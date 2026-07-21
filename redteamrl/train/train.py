import torch
from redteamrl.harness.episode import run_episode
from redteamrl.train.capture import Example
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


def rollout(tasks, attack_agent_factory, benign_agent_factory, defender_factory,
            sandbox_factory, capturing_generate, n_rollouts: int, max_turns: int = 12) -> list[Example]:
	"""Run n_rollouts episodes per task; tag every captured DEFENDER Example with that episode's
	defender_reward. The attack/benign agents are built per-spec by the caller's factories
	(scripted, for the de-risk run) — so the ONLY live, captured model is the defender. Every bit
	of within-group reward variance is therefore attributable to the defender's sampled decisions,
	which is exactly the credit-assignment we want when training the defender."""
	examples: list[Example] = []
	episode_id = 0
	for ti, spec in enumerate(tasks):
		task_rewards = []
		td = {"def_dec": 0, "def_fail": 0, "atk_act": 0, "atk_inv": 0, "ben_act": 0, "ben_inv": 0, "turns": []}
		for _ in range(n_rollouts):
			start = len(capturing_generate.buffer)
			factory = attack_agent_factory if spec.episode_type == "attack" else benign_agent_factory
			agent = factory(spec)
			sandbox = sandbox_factory()
			try:
				result = run_episode(spec, agent, defender_factory(), sandbox, max_turns=max_turns)
			finally:
				sandbox.close()

			for ex in capturing_generate.buffer[start:]:
				ex.task_id, ex.episode_id, ex.reward = spec.id, episode_id, result.defender_reward
			examples.extend(capturing_generate.buffer[start:])
			task_rewards.append(result.defender_reward)
			_tally_invalid(td, spec.episode_type, result)
			episode_id += 1
		
		# per task: rewards (all-identical = dead group; mixed = live) + WHERE invalid JSON came
		# from (agent truncating vs defender fail-closing) + avg episode length — no guessing.
		agent_inv, agent_acts = td["atk_inv"] + td["ben_inv"], td["atk_act"] + td["ben_act"]
		avg_turns = sum(td["turns"]) / max(len(td["turns"]), 1)
		print(f"    [rollout] {ti + 1}/{len(tasks)} {spec.id}: rewards={task_rewards}  "
		      f"invalid[agent {agent_inv}/{agent_acts}, defender {td['def_fail']}/{td['def_dec']}]  "
		      f"avg_turns={avg_turns:.1f}", flush=True)
	return examples

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
	learner.optimizer.zero_grad()
	total = 0.0
	ratios = []
	n = 0
	for example in examples:
		if example.advantage == 0:
			continue
		new_lp = learner.logprobs(example.prompt_ids, example.completion_ids, use_adapter=True, with_grad=True)
		old_lp = learner.logprobs(example.prompt_ids, example.completion_ids, use_adapter=True, with_grad=False).detach()
		ref_lp = learner.logprobs(example.prompt_ids, example.completion_ids, use_adapter=False, with_grad=False).detach()
		mask = torch.ones_like(new_lp)
		loss = example_loss(new_lp, old_lp, ref_lp, example.advantage, mask, beta, clip_eps)
		loss.backward()
		total += loss.item()
		ratios.append(torch.exp(new_lp - old_lp).mean().item())
		n += 1
	learner.optimizer.step()
	return {
		"loss": total / max(n, 1), "mean_ratio": sum(ratios) / max(len(ratios), 1),
		"n_live_examples": n, "n_examples": len(examples)
	}