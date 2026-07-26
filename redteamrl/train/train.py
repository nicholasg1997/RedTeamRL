from collections import Counter
from concurrent.futures import ThreadPoolExecutor

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


def rollout(tasks, attack_agent_factory, benign_agent_factory, defender_factory,
            sandbox_factory, capturing_generate, n_rollouts: int, max_turns: int = 12,
            redaction_enforcement: RedactionEnforcement = "unshielded",
            trained_side: str = "defender", post_completion_turns: int | None = None,
            max_workers: int = 1, episode_store: str | None = None,
            commit=None) -> list[Example]:
	"""Run n_rollouts episodes per task; tag every captured Example with that episode's reward for
	the side being trained.

	Exactly ONE side is live and captured per run — whichever `capturing_generate` was handed to.
	`trained_side` selects which reward tags those examples, so the same loop serves the defender
	phase and the attacker phase. Keeping the opponent frozen is what makes within-group reward
	variance attributable to the trained model; turning both sides live before the loop is proven
	makes a dead group undiagnosable.

	`max_workers` > 1 runs a task's episodes concurrently so the frozen opponent's vLLM server
	batches instead of serving one request at a time — that server is ~85% of wall clock at
	batch-1. Episode ids are derived from (task index, rollout index), so attribution does not
	depend on completion order.

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
	examples: list[Example] = []

	for ti, spec in enumerate(tasks):
		factory = attack_agent_factory if spec.episode_type == "attack" else benign_agent_factory
		td = {"def_dec": 0, "def_fail": 0, "atk_act": 0, "atk_inv": 0, "ben_act": 0, "ben_inv": 0, "turns": []}

		def run(rollout_index):
			episode_id = ti * n_rollouts + rollout_index
			if episode_id in banked:
				return banked[episode_id]
			outcome = _run_one_episode(
				spec, episode_id, factory, defender_factory,
				sandbox_factory, capturing_generate, max_turns, post_completion_turns,
				redaction_enforcement, trained_side,
			)
			if episode_store:
				write_episode(episode_store, outcome)
			return outcome

		if max_workers == 1:
			outcomes = [run(r) for r in range(n_rollouts)]
		else:
			with ThreadPoolExecutor(max_workers=max_workers) as pool:
				# Materialize in submission order: diagnostics stay deterministic even though the
				# episodes themselves finish out of order.
				outcomes = list(pool.map(run, range(n_rollouts)))

		task_rewards = []
		for outcome in outcomes:
			examples.extend(outcome["examples"])
			task_rewards.append(outcome["reward"])
			for key, value in outcome["tally"].items():
				if key == "turns":
					td["turns"].append(value)
				else:
					td[key] += value
		# Flush from THIS thread: a Modal Volume commit from a worker is not obviously safe, and
		# a task boundary is a small enough unit of loss.
		if commit is not None:
			commit()

		# per task: rewards (all-identical = dead group; mixed = live) + WHERE invalid JSON came
		# from (agent truncating vs defender fail-closing) + avg episode length — no guessing.
		agent_inv, agent_acts = td["atk_inv"] + td["ben_inv"], td["atk_act"] + td["ben_act"]
		avg_turns = sum(td["turns"]) / max(len(td["turns"]), 1)
		print(f"    [rollout] {ti + 1}/{len(tasks)} {spec.id}: rewards={task_rewards}  "
		      f"invalid[agent {agent_inv}/{agent_acts}, defender {td['def_fail']}/{td['def_dec']}]  "
		      f"avg_turns={avg_turns:.1f}", flush=True)
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

def update_step(learner, examples: list[Example], beta: float = 0.04, clip_eps: float = 0.2) -> dict:
	"""One GRPO step, weighting every episode equally regardless of trajectory length.

	Credit is assigned at episode granularity, so a ten-decision trajectory must not receive ten
	times the weight of a one-decision one. Each decision is scaled by
	``1 / (n_episodes * decisions_in_its_episode)`` and backpropagated IMMEDIATELY.

	Accumulating the losses and calling a single ``backward()`` at the end is mathematically
	identical but holds one autograd graph per live example simultaneously. At iteration 0 that
	was 350 graphs through a 4B model beside a 27B vLLM server, which OOM'd an 80GB A100.
	"""
	learner.optimizer.zero_grad()
	live = [example for example in examples if example.advantage != 0]
	decisions_per_episode = Counter(example.episode_id for example in live)
	n_episodes = len(decisions_per_episode)
	ratios = []
	kls = []
	total_loss = 0.0

	for example in live:
		new_lp = learner.logprobs(example.prompt_ids, example.completion_ids, use_adapter=True, with_grad=True)
		old_lp = learner.logprobs(example.prompt_ids, example.completion_ids, use_adapter=True, with_grad=False).detach()
		ref_lp = learner.logprobs(example.prompt_ids, example.completion_ids, use_adapter=False, with_grad=False).detach()
		mask = torch.ones_like(new_lp)
		loss = example_loss(new_lp, old_lp, ref_lp, example.advantage, mask, beta, clip_eps)
		weight = 1.0 / (n_episodes * decisions_per_episode[example.episode_id])
		(loss * weight).backward()
		total_loss += loss.item() * weight
		ratios.append(torch.exp(new_lp - old_lp).mean().item())
		# Report the same non-negative k3 estimator used by the optimization objective.
		delta = ref_lp - new_lp
		kls.append((torch.exp(delta) - delta - 1.0).mean().item())

	if live:
		trainable = [p for p in learner.model.parameters() if p.requires_grad]
		grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=float("inf")))
		learner.optimizer.step()
	else:
		grad_norm = 0.0
		total_loss = 0.0

	return {
		"loss": total_loss,
		"mean_ratio": sum(ratios) / max(len(ratios), 1),
		"mean_kl": sum(kls) / max(len(kls), 1),
		"grad_norm": grad_norm,
		"n_live_episodes": n_episodes,
		"n_live_examples": len(live),
		"n_examples": len(examples),
	}
