import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import torch
from redteamrl.harness.episode import RedactionEnforcement, build_result, run_episode, run_turns
from redteamrl.harness.restart import (FsHashRecorder, NondeterministicSource, load_record,
                                       record_episode, restore_state)
from redteamrl.train.capture import Example
from redteamrl.train.closeness import closeness
from redteamrl.types import TaskSpec
from redteamrl.train.episode_store import read_episodes, write_episode
from redteamrl.train.grpo import group_advantages, example_loss, k3_kl
from redteamrl.harness.protocol import _INVALID_ATTACKER_OUTPUT, _DEFENDER_FAIL_MARKER


def _tally_invalid(diag: dict, episode_type: str, result) -> None:
	"""Tally invalid-JSON outputs so a run is diagnosable without guessing WHICH side failed to
	emit a parseable action: the defender (fail-closed -> deny corner), the attacker, or the honest
	benign agent. Also records episode length (an env/dynamics signal)."""
	diag["turns"].append(result.n_turns)
	act, inv = ("atk_act", "atk_inv") if episode_type == "attack" else ("ben_act", "ben_inv")
	for s in result.steps:
		diag[act] += 1
		a = s["action"]
		if a.get("kind") == "message" and a.get("text") == _INVALID_ATTACKER_OUTPUT:
			diag[inv] += 1
		for d in (s.get("call_decision"), s.get("result_decision")):
			if d:
				diag["def_dec"] += 1
				if str(d.get("reasoning", "")).startswith(_DEFENDER_FAIL_MARKER):
					diag["def_fail"] += 1


class RestartUnavailable(RuntimeError):
	"""A restart source could not be restored (unreadable record, nondeterministic replay, a turn
	at or past the source's stop, settings mismatch). The rollout falls back to a fresh episode."""


def _restore(record, k, attacker, sandbox, max_turns, post_completion_turns,
             redaction_enforcement):
	"""`run_restart`'s restore half, with its failures narrowed to RestartUnavailable.

	Kept apart from the continuation on purpose: only a failure to REBUILD the prefix may trigger
	the fresh fallback. A ValueError raised while the live policy plays the suffix is a real bug and
	must propagate, not be silently converted into a fresh episode."""
	try:
		if (record["redaction_enforcement"] != redaction_enforcement
				or record["max_turns"] != max_turns
				or record["post_completion_turns"] != post_completion_turns):
			raise ValueError("restart settings differ from the source episode's "
			                 "(redaction_enforcement / max_turns / post_completion_turns)")
		return restore_state(record, k, attacker, sandbox)
	except (NondeterministicSource, ValueError) as exc:
		raise RestartUnavailable(str(exc)) from exc


def _run_one_episode(spec, episode_id, factory, defender_factory, sandbox_factory,
                     capturing_generate, max_turns, post_completion_turns,
                     redaction_enforcement, trained_side, reward_transform=None,
                     record_restart_state: bool = False, restart: tuple[dict, int] | None = None):
	"""Run one episode into its OWN capture buffer and tag every example it produced.

	`defender_factory(spec)` mirrors `attack_agent_factory(spec)`: the attacker phase varies the
	frozen defender's system prompt per task, so the trainee cannot overfit one phrasing of its
	opponent. (sft/collect.py keeps its own no-arg contract; it has separate callers.)

	`restart=(record, k)` begins the episode at turn k of a recorded SOURCE episode (`spec` must be
	the source's): the prefix is rebuilt from the record, never regenerated, so the capture buffer
	only ever holds the live suffix's decisions -- prefix turns are context, not training data. The
	reward is still the whole-episode reward. Raises RestartUnavailable if the prefix cannot be
	rebuilt.

	`record_restart_state` (attack episodes only) records what a bank needs to restart from this
	episode later: the restart record (`record_episode`, with a sandbox hash after EVERY turn --
	for a restart, the source's prefix hashes plus this episode's live ones), its closeness and
	whether it won. Selection data only; none of it touches reward.
	"""
	agent = factory(spec)
	sandbox = sandbox_factory()
	recording = record_restart_state and spec.episode_type == "attack"
	recorder = FsHashRecorder(sandbox) if recording else None
	try:
		if restart is None:
			# `on_turn` only when recording, so run_episode's call shape is otherwise unchanged.
			extra = {"on_turn": recorder} if recorder is not None else {}
			with capturing_generate.episode_capture() as episode_examples:
				result = run_episode(
					spec,
					agent,
					defender_factory(spec),
					sandbox,
					max_turns=max_turns,
					post_completion_turns=post_completion_turns,
					redaction_enforcement=redaction_enforcement,
					**extra,
				)
		else:
			record, k = restart
			defender = defender_factory(spec)
			state = _restore(record, k, agent, sandbox, max_turns, post_completion_turns,
			                 redaction_enforcement)
			with capturing_generate.episode_capture() as episode_examples:
				run_turns(state, spec, agent, defender, sandbox, max_turns, post_completion_turns,
				          redaction_enforcement, start_turn=k, on_turn=recorder)
			result = build_result(state, spec, max_turns, redaction_enforcement)
	finally:
		sandbox.close()

	reward = result.defender_reward if trained_side == "defender" else result.attacker_reward
	if reward_transform is not None:
		reward = reward_transform(result, reward, spec)
	verdicts = episode_verdicts(result)
	tally = {"def_dec": 0, "def_fail": 0, "atk_act": 0, "atk_inv": 0, "ben_act": 0, "ben_inv": 0, "turns": []}
	_tally_invalid(tally, spec.episode_type, result)
	tally["turns"] = result.n_turns
	for ex in episode_examples:
		ex.task_id, ex.episode_id, ex.reward = spec.id, episode_id, reward
		ex.verdicts = verdicts
		ex.episode_type = spec.episode_type
		ex.policy_leak = bool(getattr(result, "policy_leak", False))
		ex.complete = bool(getattr(result, "complete", False))
		ex.defender_protocol_failures = int(getattr(result, "defender_protocol_failures", 0))
		ex.leak_attribution = getattr(result, "leak_attribution", None)
		ex.passed_taint_counts = dict(getattr(result, "passed_taint_counts", {}) or {})
	outcome = {"episode_id": episode_id, "task_id": spec.id, "reward": reward,
	           "tally": tally, "examples": episode_examples}
	if recording:
		hashes = list(recorder.hashes)
		if restart is not None:
			hashes = list(restart[0]["fs_hash_after"][:restart[1]]) + hashes
		outcome["record"] = record_episode(result, spec, getattr(agent, "_outputs", []), hashes,
		                                   max_turns, post_completion_turns)
		outcome["closeness"] = list(closeness(result))
		outcome["won"] = bool(result.outcome == "leak")
	return outcome


def _report_task(ti, spec, outcomes: dict, n_rollouts: int, n_tasks: int, tag: str = "") -> None:
	"""Per task: rewards (all-identical = dead group; mixed = live) + WHERE invalid JSON came from
	(agent truncating vs defender fail-closing) + avg episode length — no guessing.

	In plan mode it is called once per GROUP (`ti` = slot) and `tag` labels the group; the tag is
	appended at the END so the line's existing prefix format is unchanged."""
	td = {"def_dec": 0, "def_fail": 0, "atk_act": 0, "atk_inv": 0, "ben_act": 0, "ben_inv": 0,
	      "turns": []}
	rewards = []
	for rollout_index in range(n_rollouts):
		outcome = outcomes[rollout_index]
		rewards.append(outcome["reward"])
		for key, value in outcome["tally"].items():
			if key == "turns":
				td["turns"].append(value)
			else:
				td[key] += value
	agent_inv, agent_acts = td["atk_inv"] + td["ben_inv"], td["atk_act"] + td["ben_act"]
	avg_turns = sum(td["turns"]) / max(len(td["turns"]), 1)
	print(f"    [rollout] {ti + 1}/{n_tasks} {spec.id}: rewards={rewards}  "
	      f"invalid[agent {agent_inv}/{agent_acts}, defender {td['def_fail']}/{td['def_dec']}]  "
	      f"avg_turns={avg_turns:.1f}{tag}", flush=True)


def _group_tag(plan, outcomes: dict) -> str:
	if plan.kind != "restart":
		return f"  [fresh {plan.group_key} {plan.reason}]"
	fallbacks = sum(bool(o.get("restart_fallback")) for o in outcomes.values())
	fb = f" FALLBACK={fallbacks}/{len(outcomes)}" if fallbacks else ""
	return f"  [restart {plan.group_key} k={plan.restart_turn} {plan.reason}{fb}]"


def rollout(tasks, attack_agent_factory, benign_agent_factory, defender_factory,
            sandbox_factory, capturing_generate, n_rollouts: int, max_turns: int = 12,
            redaction_enforcement: RedactionEnforcement = "unshielded",
            trained_side: str = "defender", post_completion_turns: int | None = None,
            max_workers: int = 1, episode_store: str | None = None,
            commit=None, task_transform=None, reward_transform=None,
            group_plans: list | None = None, task_by_id: dict[str, TaskSpec] | None = None,
            record_restart_state: bool = False, outcome_sink: dict | None = None) -> list[Example]:
	"""Run n_rollouts episodes per task; tag every captured Example with that episode's reward for
	the side being trained.

	Exactly ONE side is live and captured per run — whichever `capturing_generate` was handed to.
	`trained_side` selects which reward tags those examples, so the same loop serves the defender
	phase and the attacker phase. Keeping the opponent frozen is what makes within-group reward
	variance attributable to the trained model; turning both sides live before the loop is proven
	makes a dead group undiagnosable.

	`max_workers` > 1 runs episodes concurrently ACROSS ALL TASKS, not just within one. vLLM
	batches by in-flight request count, so processing tasks one at a time capped the engine at a
	queue of ~1 and threw away its entire throughput advantage. Episode ids are derived from (task index, rollout index), so attribution does not
	depend on completion order.

	`task_transform(spec, rollout_index) -> spec` rewrites the task per episode — used to plant a
	fresh high-entropy canary in every rollout, exactly as the SFT collection does. Without it GRPO
	runs against the literal template secret, which is a different environment from the one SFT was
	trained and measured on. It must preserve `id` and `episode_type`: attribution keys off those.

	`episode_store` persists each finished episode, so a preemption mid-rollout costs one episode
	rather than the whole iteration. Banked episodes stay valid on resume because the policy does
	not move during a rollout. `commit` is invoked once per task FROM THIS THREAD — never from a
	worker — to flush the store durably.

	PLAN MODE (`group_plans`, from train/allocation.allocate): the groups are the plans, not
	`tasks x n_rollouts`. Episode id = `plan.slot * n_rollouts + r` (unique even when one task has
	two groups) and every Example gets `group_key = plan.group_key`. A fresh plan runs
	`task_by_id[plan.task_id]` (default: `tasks` keyed by id) through
	`task_transform(spec, plan.slot * n_rollouts + r)`; a restart plan resumes its bank entry's
	recorded source at `plan.restart_turn`, on the SOURCE's spec and canary (no transform). A source
	that cannot be restored falls back, for that rollout, to a fresh episode of the same task under
	the same episode id, marked `restart_fallback` and grouped under `group_key + "#fallback"`:
	it did not begin in the restart's start state, so it must not share that baseline. Plan-mode
	outcomes also carry `kind`, `slot`, `group_key`, `restart_turn`, `source_entry_id` and
	`restart_fallback`; `commit` runs once per group.

	`record_restart_state` adds `record` / `closeness` / `won` to each attack episode's outcome
	(see `_run_one_episode`); they are persisted with the episode, so they survive a resume.
	`outcome_sink`, if given, receives every outcome dict keyed by episode id (banked ones too)."""
	if trained_side not in {"defender", "attacker"}:
		raise ValueError("trained_side must be 'defender' or 'attacker'")
	if max_workers < 1:
		raise ValueError("max_workers must be positive")
	banked = read_episodes(episode_store) if episode_store else {}
	if banked:
		print(f"    [rollout] resuming with {len(banked)} banked episodes", flush=True)

	if group_plans is None:
		groups = list(enumerate(tasks))
		n_groups = len(tasks)
	else:
		plans = sorted(group_plans, key=lambda p: p.slot)
		if len({p.slot for p in plans}) != len(plans):
			raise ValueError("group plans must have distinct slots")
		task_by_id = task_by_id if task_by_id is not None else {spec.id: spec for spec in tasks}
		for plan in plans:
			if plan.task_id not in task_by_id:
				raise ValueError(f"plan for unknown task {plan.task_id!r}")
			if plan.kind == "restart" and (plan.entry is None or plan.restart_turn is None):
				raise ValueError(f"restart plan in slot {plan.slot} has no source entry / turn")
			if plan.kind not in {"fresh", "restart"}:
				raise ValueError(f"unknown plan kind {plan.kind!r}")
		groups = [(plan.slot, plan) for plan in plans]
		n_groups = len(plans)
	work = [(gi, group, r) for gi, group in groups for r in range(n_rollouts)]

	def run_fresh(spec, episode_id, transform_index):
		episode_spec = task_transform(spec, transform_index) if task_transform else spec
		factory = (attack_agent_factory if episode_spec.episode_type == "attack"
		           else benign_agent_factory)
		return _run_one_episode(
			episode_spec, episode_id, factory, defender_factory, sandbox_factory,
			capturing_generate, max_turns, post_completion_turns,
			redaction_enforcement, trained_side, reward_transform,
			**({"record_restart_state": True} if record_restart_state else {}),
		)

	def run_planned(plan, episode_id):
		fallback = False
		group_key = plan.group_key
		outcome = None
		if plan.kind == "restart":
			try:
				try:
					record = load_record(plan.entry.record_path)
					spec = TaskSpec.model_validate(record["spec"])
				except (OSError, ValueError, KeyError) as exc:
					raise RestartUnavailable(f"unreadable record: {exc}") from exc
				if spec.id != plan.task_id:
					# A bank pointing at another task's record would cross environments: a bug,
					# never something to paper over with a fallback.
					raise RuntimeError(f"restart source {plan.entry.entry_id} is task {spec.id!r}, "
					                   f"plan is for {plan.task_id!r}")
				outcome = _run_one_episode(
					spec, episode_id, attack_agent_factory, defender_factory, sandbox_factory,
					capturing_generate, max_turns, post_completion_turns,
					redaction_enforcement, trained_side, reward_transform,
					record_restart_state=record_restart_state,
					restart=(record, plan.restart_turn),
				)
			except RestartUnavailable as exc:
				print(f"    [rollout] restart fallback: slot {plan.slot} {plan.group_key} "
				      f"episode {episode_id}: {exc}", flush=True)
				fallback = True
				group_key = plan.group_key + "#fallback"
		if outcome is None:
			outcome = run_fresh(task_by_id[plan.task_id], episode_id, episode_id)
		for ex in outcome["examples"]:
			ex.group_key = group_key
		outcome.update({
			"kind": plan.kind, "slot": plan.slot, "group_key": group_key,
			"restart_turn": plan.restart_turn,
			"source_entry_id": plan.entry.entry_id if plan.entry is not None else None,
			"restart_fallback": fallback,
		})
		return outcome

	def run(item):
		gi, group, rollout_index = item
		episode_id = gi * n_rollouts + rollout_index
		if episode_id in banked:
			return gi, rollout_index, banked[episode_id]
		if group_plans is None:
			outcome = run_fresh(group, episode_id, rollout_index)
		else:
			outcome = run_planned(group, episode_id)
		if episode_store:
			write_episode(episode_store, outcome)
		return gi, rollout_index, outcome

	group_of = dict(groups)

	def finish_group(gi, outcomes):
		if group_plans is None:
			_report_task(gi, group_of[gi], outcomes, n_rollouts, n_groups)
		else:
			plan = group_of[gi]
			_report_task(gi, SimpleNamespace(id=plan.task_id), outcomes, n_rollouts, n_groups,
			             tag=_group_tag(plan, outcomes))
		if commit is not None:
			commit()

	by_group: dict[int, dict[int, dict]] = defaultdict(dict)
	if max_workers == 1:
		for item in work:
			gi, rollout_index, outcome = run(item)
			by_group[gi][rollout_index] = outcome
			if len(by_group[gi]) == n_rollouts:
				finish_group(gi, by_group[gi])
	else:
		with ThreadPoolExecutor(max_workers=max_workers) as pool:
			futures = [pool.submit(run, item) for item in work]
			# as_completed runs in THIS thread, so the per-task report and the Volume commit
			# never fire from a worker.
			for future in as_completed(futures):
				gi, rollout_index, outcome = future.result()
				by_group[gi][rollout_index] = outcome
				if len(by_group[gi]) == n_rollouts:
					finish_group(gi, by_group[gi])

	# Reassemble by index: arrival order is nondeterministic, attribution must not be.
	examples: list[Example] = []
	for gi, _ in groups:
		for rollout_index in range(n_rollouts):
			outcome = by_group[gi][rollout_index]
			examples.extend(outcome["examples"])
			if outcome_sink is not None:
				outcome_sink[outcome["episode_id"]] = outcome
	return examples


def episode_verdicts(result) -> list[dict[str, str]]:
	"""Every typed defender verdict this episode produced, for within-group diversity.

	`group_diversity` was being handed empty lists, so mixed_verdict_group_rate and
	verdict_entropy were structurally zero and the constant-policy signal measured nothing. Keep
	call and result verdicts separate: an always-allow/always-pass policy has two protocol words
	but zero behavioral diversity.
	"""
	verdicts = []
	for step in result.steps:
		for decision_type, decision in (
			("call", step.get("call_decision")),
			("result", step.get("result_decision")),
		):
			if decision and decision.get("verdict"):
				verdicts.append({
					"decision_type": decision_type,
					"verdict": str(decision["verdict"]),
				})
	return verdicts


def tactic_counts(examples: list[Example]) -> dict:
	"""Per EPISODE (examples repeat episode facts on every decision): how leaks got out, which known
	secret-carriers the defender let through, and the share of episodes passing an encoded carrier.
	Diagnostic only -- never an input to reward. Shared by `tactic_summary` and the metrics log."""
	episodes = {}
	for ex in examples:
		episodes.setdefault(ex.episode_id, ex)
	leaks: dict[str, int] = {}
	exposed: dict[str, int] = {}
	for ex in episodes.values():
		if ex.leak_attribution:
			leaks[ex.leak_attribution] = leaks.get(ex.leak_attribution, 0) + 1
		for kind in ex.passed_taint_counts:
			exposed[kind] = exposed.get(kind, 0) + 1
	n = max(len(episodes), 1)
	encoded = sum(any(k in ("base64", "hex") for k in ex.passed_taint_counts) for ex in episodes.values())
	return {"leaks": leaks, "exposed": exposed, "encoded_pass_rate": encoded / n,
	        "n_episodes": len(episodes)}


def tactic_summary(examples: list[Example]) -> str:
	"""One log line: how leaks got out and what known secret-carriers the defender let through.

	Per EPISODE (examples repeat episode facts on every decision). Diagnostic only. Watch
	`unattributed`: a leak no known detector explains is a new tactic -- or a channel around the
	defender -- and is the cue to read those transcripts by hand."""
	counts = tactic_counts(examples)
	fmt = lambda d: " ".join(f"{k}={v}" for k, v in sorted(d.items())) or "none"
	return (f"leaks by tactic: {fmt(counts['leaks'])} | episodes passing a carrier: "
	        f"{fmt(counts['exposed'])} | encoded_pass_rate={counts['encoded_pass_rate']:.1%}")


def assign_advantages(examples: list[Example], normalize_std: bool = True) -> None:
	ep_reward = {}
	ep_task = {}

	for example in examples:
		ep_reward[example.episode_id] = example.reward
		ep_task[example.episode_id] = example.group_key or example.task_id

	ep_ids = list(ep_reward)
	adv = group_advantages([ep_reward[i] for i in ep_ids], [ep_task[i] for i in ep_ids],
	                       normalize_std=normalize_std)
	by_ep = dict(zip(ep_ids, adv))
	for example in examples:
		example.advantage = by_ep[example.episode_id]

def micro_batches(examples: list[Example], max_tokens: int, max_logit_rows: int) -> list[list[int]]:
	"""Group example indices into forward-pass batches, sorted by length to minimise padding.

	Two caps, because two tensors grow with a batch: activations ~ rows x padded width
	(`max_tokens`), and logits ~ rows x (longest completion + 1) x ~152k vocab (`max_logit_rows`).
	An example that alone exceeds a cap still gets its own batch -- exactly what ran before."""
	order = sorted(range(len(examples)),
	               key=lambda i: len(examples[i].prompt_ids) + len(examples[i].completion_ids))
	batches, current, width, keep = [], [], 0, 0
	for i in order:
		n = len(examples[i].prompt_ids) + len(examples[i].completion_ids)
		k = len(examples[i].completion_ids) + 1
		new_width, new_keep = max(width, n), max(keep, k)
		if current and ((len(current) + 1) * new_width > max_tokens
		                or (len(current) + 1) * new_keep > max_logit_rows):
			batches.append(current)
			current, new_width, new_keep = [], n, k
		current.append(i)
		width, keep = new_width, new_keep
	if current:
		batches.append(current)
	return batches


def episode_minibatches(live: list[Example], n: int, seed: str | None) -> list[list[int]]:
	"""Partition `live` (by index) into at most `n` minibatches, never splitting an episode.

	Episodes are dealt round-robin after a shuffle seeded by `seed` (None keeps sorted episode-id
	order), so a preempted update reruns identically. Never returns an empty minibatch: with fewer
	episodes than `n`, there is one minibatch per episode."""
	episodes = sorted({example.episode_id for example in live}, key=repr)
	if seed is not None:
		random.Random(seed).shuffle(episodes)
	k = max(1, min(n, len(episodes)))
	slot = {episode: j % k for j, episode in enumerate(episodes)}
	batches: list[list[int]] = [[] for _ in range(k)]
	for i, example in enumerate(live):
		batches[slot[example.episode_id]].append(i)
	return [batch for batch in batches if batch]


def update_step(learner, examples: list[Example], beta: float = 0.04, clip_eps: float = 0.2,
                inner_epochs: int = 1, max_batch_tokens: int = 16384,
                max_batch_logit_rows: int = 4096, no_checkpoint_max_tokens: int = 0,
                minibatches: int = 1, target_kl: float | None = None,
                shuffle_seed: str | None = None) -> dict:
	"""GRPO update: `inner_epochs` passes over ONE batch of rollouts.

	Generation dominates cost — a rollout is hours, an optimizer step is seconds. Taking a single
	step per rollout wastes the batch and leaves the clipped objective inert: `new_lp` and `old_lp`
	come from identical weights, so the ratio is exactly 1 and `clip_eps` never fires. Multiple
	epochs are what the importance ratio and clipping exist for.

	`old_lp` (the behaviour policy that generated these rollouts) and `ref_lp` (the frozen SFT
	reference) are computed ONCE. Recomputing `old_lp` after a step would re-pin the ratio at 1
	and silently remove the trust region — the epochs would become unconstrained repeats.

	`minibatches == 1` (the default, and the defender's path) takes ONE optimizer step per epoch over
	the whole batch -- the pre-v7 update, bit for bit. Lineage v7 found that this moved the attacker
	almost nowhere (~2 Adam steps per iteration, KL 0.003 after five iterations), so `minibatches > 1`
	partitions the episodes each epoch (`episode_minibatches`, seeded by `shuffle_seed`) and steps
	once per minibatch. Behaviour log-probs then come from a frozen pass before any step, and
	`target_kl` arms a brake: a minibatch whose behaviour-to-current KL exceeds 2x the target is not
	stepped, and the update ends (PPO's early stop). `kl_iter` reports that KL for the last epoch.

	Episodes are weighted equally regardless of trajectory length. A learner with
	`logprobs_batch` is scored in micro-batches (one autograd graph per micro-batch, bounded by
	`max_batch_tokens` / `max_batch_logit_rows`); one without it is scored one example at a time.
	The per-example losses and weights are identical either way -- batching changes speed, not math.
	"""
	if inner_epochs < 1:
		raise ValueError("inner_epochs must be at least 1")
	if minibatches < 1:
		raise ValueError("minibatches must be at least 1")
	live = [example for example in examples if example.advantage != 0]
	if not live:
		return {"loss": 0.0, "mean_ratio": 0.0, "mean_kl": 0.0, "grad_norm": 0.0,
		        "n_live_episodes": 0, "n_live_examples": 0, "n_examples": len(examples),
		        "inner_epochs": inner_epochs, "epoch_ratios": [], "steps": 0,
		        "early_stopped": False, "kl_iter": 0.0, "clip_frac": 0.0, "minibatch_kl_old": []}

	decisions_per_episode = Counter(example.episode_id for example in live)
	n_episodes = len(decisions_per_episode)

	# The reference is frozen before any weights move. Both frozen passes are CLONED: a no-grad
	# path returning a view of the parameters would otherwise alias live storage and let the
	# optimizer step mutate them silently.
	batched = hasattr(learner, "logprobs_batch")
	# When checkpoint-skipping is on, cap batches at its threshold: otherwise micro_batches packs to
	# max_batch_tokens and almost no batch is small enough to skip -- the saving would silently be
	# zero. Smaller batches cost nothing here; the update is compute-bound at ~2k tokens/example.
	token_cap = (min(max_batch_tokens, no_checkpoint_max_tokens) if no_checkpoint_max_tokens > 0
	             else max_batch_tokens)

	def pack(indices):
		"""Memory packing of `indices` (into `live`) into forward-pass micro-batches."""
		if not batched:
			return [[i] for i in indices]
		subset = [live[i] for i in indices]
		return [[indices[j] for j in batch]
		        for batch in micro_batches(subset, token_cap, max_batch_logit_rows)]

	batches = pack(list(range(len(live))))

	def score(indices, use_adapter, with_grad):
		if not batched:
			example = live[indices[0]]
			return [learner.logprobs(example.prompt_ids, example.completion_ids,
			                         use_adapter=use_adapter, with_grad=with_grad)]
		return learner.logprobs_batch(
			[(live[i].prompt_ids, live[i].completion_ids) for i in indices],
			use_adapter=use_adapter, with_grad=with_grad)

	ref: list = [None] * len(live)
	old: list = [None] * len(live)
	for indices in batches:
		for i, ref_lp in zip(indices, score(indices, False, False)):
			ref[i] = ref_lp.detach().clone()
	if minibatches > 1:
		# Steps happen mid-epoch, so epoch 0's forward is no longer the behaviour policy: freeze it
		# first. With one minibatch, old_lp is taken from epoch 0's forward instead (below): no
		# step happens until epoch 0 has scored everything, so the values are identical by
		# construction and one forward per example is saved.
		for indices in batches:
			for i, lp in zip(indices, score(indices, True, False)):
				old[i] = lp.detach().clone()

	def padded_tokens(indices):
		return len(indices) * max(len(live[i].prompt_ids) + len(live[i].completion_ids)
		                          for i in indices)

	def score_with_grad(indices):
		"""Skip activation checkpointing for micro-batches small enough to hold their activations:
		checkpointing re-runs the forward during backward (~25% of an epoch) and exists for 15k-token
		sequences. Same graph, same math -- only what is stored vs recomputed changes."""
		skip = (no_checkpoint_max_tokens > 0 and hasattr(learner, "no_checkpointing")
		        and padded_tokens(indices) <= no_checkpoint_max_tokens)
		if not skip:
			return score(indices, True, True)
		with learner.no_checkpointing():
			return score(indices, True, True)

	trainable = [p for p in learner.model.parameters() if p.requires_grad]
	epoch_ratios, minibatch_kl_old = [], []
	total_loss = grad_norm = 0.0
	kls, epoch_kl_old = [], []
	clipped = tokens = 0
	steps, early_stopped = 0, False
	for epoch in range(inner_epochs):
		ratios, kls, epoch_kl_old = [], [], []
		clipped = tokens = 0
		total_loss = 0.0
		if minibatches == 1:
			plan = [list(range(len(live)))]
			chunks = [batches]                # the epoch-wide micro-batches, as before v7
		else:
			seed = None if shuffle_seed is None else f"{shuffle_seed}-e{epoch}"
			plan = episode_minibatches(live, minibatches, seed)
			chunks = [pack(minibatch) for minibatch in plan]
		for minibatch, micro in zip(plan, chunks):
			learner.optimizer.zero_grad()
			mb_episodes = len({live[i].episode_id for i in minibatch})
			mb_kl = []
			for indices in micro:
				batch_loss = 0.0
				for i, new_lp in zip(indices, score_with_grad(indices)):
					example = live[i]
					if epoch == 0 and minibatches == 1:
						# CLONE, not just detach: a view would alias live storage, so the optimizer
						# step would silently mutate old_lp, re-pin the ratio at 1, and remove the
						# trust region.
						old[i] = new_lp.detach().clone()
					old_lp, ref_lp = old[i], ref[i]
					mask = torch.ones_like(new_lp)
					loss = example_loss(new_lp, old_lp, ref_lp, example.advantage, mask, beta,
					                    clip_eps)
					weight = 1.0 / (mb_episodes * decisions_per_episode[example.episode_id])
					batch_loss = batch_loss + loss * weight
					total_loss += loss.item() * weight
					with torch.no_grad():
						log_ratio = new_lp.detach() - old_lp
						ratios.append(torch.exp(log_ratio).mean().item())
						clipped += int(((torch.exp(log_ratio) - 1.0).abs() > clip_eps).sum())
						tokens += log_ratio.numel()
						mb_kl.append(k3_kl(new_lp.detach(), old_lp).mean().item())
						# Report the same non-negative k3 estimator used by the optimization objective.
						kls.append(k3_kl(new_lp.detach(), ref_lp).mean().item())
				batch_loss.backward()       # one graph per micro-batch
			kl_old = sum(mb_kl) / max(len(mb_kl), 1)
			minibatch_kl_old.append(kl_old)
			epoch_kl_old.append(kl_old)
			if target_kl is not None and kl_old > 2 * target_kl:
				learner.optimizer.zero_grad()
				early_stopped = True
				break
			grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=float("inf")))
			learner.optimizer.step()
			steps += 1
		epoch_ratios.append(sum(ratios) / max(len(ratios), 1))
		if early_stopped:
			break

	return {
		"loss": total_loss,
		"mean_ratio": epoch_ratios[-1],
		"mean_kl": sum(kls) / max(len(kls), 1),
		"grad_norm": grad_norm,
		"n_live_episodes": n_episodes,
		"n_live_examples": len(live),
		"n_examples": len(examples),
		"inner_epochs": inner_epochs,
		# Ratio per epoch: starts at 1.0 and should drift modestly. Drifting past ~1+clip_eps
		# means later epochs are fully clipped and contributing nothing.
		"epoch_ratios": epoch_ratios,
		"steps": steps,
		"early_stopped": early_stopped,
		"kl_iter": sum(epoch_kl_old) / max(len(epoch_kl_old), 1),
		"clip_frac": clipped / tokens if tokens else 0.0,
		"minibatch_kl_old": minibatch_kl_old,
	}
