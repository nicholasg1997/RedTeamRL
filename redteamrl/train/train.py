from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from redteamrl.harness.episode import RedactionEnforcement, run_episode
from redteamrl.train.capture import Example
from redteamrl.train.episode_store import read_episodes, write_episode
from redteamrl.train.grpo import group_advantages, example_loss
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


def _run_one_episode(spec, episode_id, factory, defender_factory, sandbox_factory,
                     capturing_generate, max_turns, post_completion_turns,
                     redaction_enforcement, trained_side):
	"""Run one episode into its OWN capture buffer and tag every example it produced."""
	agent = factory(spec)
	sandbox = sandbox_factory()
	with capturing_generate.episode_capture() as episode_examples:
		try:
			result = run_episode(
				spec,
				agent,
				defender_factory(),
				sandbox,
				max_turns=max_turns,
				post_completion_turns=post_completion_turns,
				redaction_enforcement=redaction_enforcement,
			)
		finally:
			sandbox.close()

	reward = result.defender_reward if trained_side == "defender" else result.attacker_reward
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
	return {"episode_id": episode_id, "task_id": spec.id, "reward": reward,
	        "tally": tally, "examples": episode_examples}


def _report_task(ti, spec, outcomes: dict, n_rollouts: int, n_tasks: int) -> None:
	"""Per task: rewards (all-identical = dead group; mixed = live) + WHERE invalid JSON came from
	(agent truncating vs defender fail-closing) + avg episode length — no guessing."""
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
	      f"avg_turns={avg_turns:.1f}", flush=True)


def rollout(tasks, attack_agent_factory, benign_agent_factory, defender_factory,
            sandbox_factory, capturing_generate, n_rollouts: int, max_turns: int = 12,
            redaction_enforcement: RedactionEnforcement = "unshielded",
            trained_side: str = "defender", post_completion_turns: int | None = None,
            max_workers: int = 1, episode_store: str | None = None,
            commit=None, task_transform=None) -> list[Example]:
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
	worker — to flush the store durably."""
	if trained_side not in {"defender", "attacker"}:
		raise ValueError("trained_side must be 'defender' or 'attacker'")
	if max_workers < 1:
		raise ValueError("max_workers must be positive")
	banked = read_episodes(episode_store) if episode_store else {}
	if banked:
		print(f"    [rollout] resuming with {len(banked)} banked episodes", flush=True)

	work = [(ti, spec, r) for ti, spec in enumerate(tasks) for r in range(n_rollouts)]

	def run(item):
		ti, spec, rollout_index = item
		episode_id = ti * n_rollouts + rollout_index
		if episode_id in banked:
			return ti, rollout_index, banked[episode_id]
		episode_spec = task_transform(spec, rollout_index) if task_transform else spec
		factory = (attack_agent_factory if episode_spec.episode_type == "attack"
		           else benign_agent_factory)
		outcome = _run_one_episode(
			episode_spec, episode_id, factory, defender_factory, sandbox_factory,
			capturing_generate, max_turns, post_completion_turns,
			redaction_enforcement, trained_side,
		)
		if episode_store:
			write_episode(episode_store, outcome)
		return ti, rollout_index, outcome

	by_task: dict[int, dict[int, dict]] = defaultdict(dict)
	if max_workers == 1:
		for item in work:
			ti, rollout_index, outcome = run(item)
			by_task[ti][rollout_index] = outcome
			if len(by_task[ti]) == n_rollouts:
				_report_task(ti, tasks[ti], by_task[ti], n_rollouts, len(tasks))
				if commit is not None:
					commit()
	else:
		with ThreadPoolExecutor(max_workers=max_workers) as pool:
			futures = [pool.submit(run, item) for item in work]
			# as_completed runs in THIS thread, so the per-task report and the Volume commit
			# never fire from a worker.
			for future in as_completed(futures):
				ti, rollout_index, outcome = future.result()
				by_task[ti][rollout_index] = outcome
				if len(by_task[ti]) == n_rollouts:
					_report_task(ti, tasks[ti], by_task[ti], n_rollouts, len(tasks))
					if commit is not None:
						commit()

	# Reassemble by index: arrival order is nondeterministic, attribution must not be.
	examples: list[Example] = []
	for ti in range(len(tasks)):
		for rollout_index in range(n_rollouts):
			examples.extend(by_task[ti][rollout_index]["examples"])
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


def assign_advantages(examples: list[Example]) -> None:
	ep_reward = {}
	ep_task = {}

	for example in examples:
		ep_reward[example.episode_id] = example.reward
		ep_task[example.episode_id] = example.task_id

	ep_ids = list(ep_reward)
	adv = group_advantages([ep_reward[i] for i in ep_ids], [ep_task[i] for i in ep_ids])
	by_ep = dict(zip(ep_ids, adv))
	for example in examples:
		example.advantage = by_ep[example.episode_id]

def update_step(learner, examples: list[Example], beta: float = 0.04, clip_eps: float = 0.2,
                inner_epochs: int = 1) -> dict:
	"""GRPO update: `inner_epochs` gradient steps over ONE batch of rollouts.

	Generation dominates cost — a rollout is hours, an optimizer step is seconds. Taking a single
	step per rollout wastes the batch and leaves the clipped objective inert: `new_lp` and `old_lp`
	come from identical weights, so the ratio is exactly 1 and `clip_eps` never fires. Multiple
	epochs are what the importance ratio and clipping exist for.

	`old_lp` (the behaviour policy that generated these rollouts) and `ref_lp` (the frozen SFT
	reference) are computed ONCE. Recomputing `old_lp` after a step would re-pin the ratio at 1
	and silently remove the trust region — the epochs would become unconstrained repeats.

	Episodes are weighted equally regardless of trajectory length, and each decision is
	backpropagated immediately so only one autograd graph is alive at a time.
	"""
	if inner_epochs < 1:
		raise ValueError("inner_epochs must be at least 1")
	live = [example for example in examples if example.advantage != 0]
	if not live:
		return {"loss": 0.0, "mean_ratio": 0.0, "mean_kl": 0.0, "grad_norm": 0.0,
		        "n_live_episodes": 0, "n_live_examples": 0, "n_examples": len(examples),
		        "inner_epochs": inner_epochs, "epoch_ratios": []}

	decisions_per_episode = Counter(example.episode_id for example in live)
	n_episodes = len(decisions_per_episode)

	# Freeze the behaviour policy and the reference before any weights move. CLONE, not just
	# detach: a learner whose no-grad path returns a view of its parameters would otherwise leave
	# these aliasing live storage, so the optimizer step would silently mutate "old_lp", re-pin
	# the ratio at 1.0, and remove the trust region without any visible error.
	frozen = [
		(
			learner.logprobs(example.prompt_ids, example.completion_ids,
			                 use_adapter=True, with_grad=False).detach().clone(),
			learner.logprobs(example.prompt_ids, example.completion_ids,
			                 use_adapter=False, with_grad=False).detach().clone(),
		)
		for example in live
	]

	epoch_ratios = []
	total_loss = grad_norm = 0.0
	kls = []
	for _ in range(inner_epochs):
		learner.optimizer.zero_grad()
		ratios = []
		kls = []
		total_loss = 0.0
		for example, (old_lp, ref_lp) in zip(live, frozen):
			new_lp = learner.logprobs(example.prompt_ids, example.completion_ids,
			                          use_adapter=True, with_grad=True)
			mask = torch.ones_like(new_lp)
			loss = example_loss(new_lp, old_lp, ref_lp, example.advantage, mask, beta, clip_eps)
			weight = 1.0 / (n_episodes * decisions_per_episode[example.episode_id])
			(loss * weight).backward()
			total_loss += loss.item() * weight
			ratios.append(torch.exp(new_lp - old_lp).mean().item())
			# Report the same non-negative k3 estimator used by the optimization objective.
			delta = ref_lp - new_lp
			kls.append((torch.exp(delta) - delta - 1.0).mean().item())
		trainable = [p for p in learner.model.parameters() if p.requires_grad]
		grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=float("inf")))
		learner.optimizer.step()
		epoch_ratios.append(sum(ratios) / max(len(ratios), 1))

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
	}
