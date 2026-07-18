import torch
from redteamrl.harness.episode import run_episode
from redteamrl.train.capture import Example
from redteamrl.train.grpo import group_advantages, example_loss


def rollout(tasks, attacker_factory, defender_factory, sandbox_factory,
            capturing_generate, n_rollouts: int) -> list[Example]:
	examples: list[Example] = []
	episode_id = 0
	for spec in tasks:
		for _ in range(n_rollouts):
			start = len(capturing_generate.buffer)
			if spec.episode_type == "attack":
				agent, turns = attacker_factory(spec.goal), 8
			else:
				from redteamrl.policies.scripted import ScriptedClient
				agent, turns = ScriptedClient(spec.client_actions), len(spec.client_actions)
			sandbox = sandbox_factory()
			try:
				result = run_episode(spec, agent, defender_factory(), sandbox, max_turns=turns)
			finally:
				sandbox.close()

			for ex in capturing_generate.buffer[start:]:
				ex.task_id, ex.episode_id, ex.reward = spec.id, episode_id, result.defender_reward
			examples.extend(capturing_generate.buffer[start:])
			episode_id += 1
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