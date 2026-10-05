from collections import defaultdict
from statistics import mean, pstdev
import torch

def group_advantages(rewards: list[float],
                     task_ids: list[str],
                     eps: float = 1e-4) -> list[float]:

	idx = defaultdict(list)

	for i, tid in enumerate(task_ids):
		idx[tid].append(i)

	adv = [0.0] * len(rewards)
	for t, ids in idx.items():
		vals = [rewards[i] for i in ids]
		mu = mean(vals)
		sd = pstdev(vals)
		for i in ids:
			adv[i] = 0.0 if sd < 1e-12 else (rewards[i] - mu) / (sd + eps)

	return adv

def k3_kl(new_lp: torch.Tensor, ref_lp: torch.Tensor) -> torch.Tensor:

	delta = ref_lp - new_lp
	return torch.exp(delta) - delta - 1.0

def example_loss(new_lp: torch.Tensor, old_lp: torch.Tensor,
          ref_lp: torch.Tensor, advantage: float, mask: torch.Tensor,
          beta: float = 0.04, clip_eps: float = 0.2) -> torch.Tensor:

	ratio = torch.exp(new_lp - old_lp)
	unclipped = ratio * advantage
	clipped = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantage
	surrogate = torch.minimum(unclipped, clipped)
	kl = k3_kl(new_lp, ref_lp)
	per_token = -(surrogate - beta * kl)
	return (per_token * mask).sum() / mask.sum().clamp(min=1.0)