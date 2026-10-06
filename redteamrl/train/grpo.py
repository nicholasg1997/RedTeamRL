from collections import defaultdict
from statistics import mean, pstdev
import torch

def group_advantages(rewards: list[float],
                     task_ids: list[str],
                     eps: float = 1e-4,
                     normalize_std: bool = True) -> list[float]:
	"""Within-group advantage: (r - mean), divided by the group's std unless normalize_std=False.

	The std division rescales every live group to unit spread, so a group whose rewards differ by
	ONE repeat-penalty step (-0.30 vs -0.35, std 0.02) pushes as hard as a win-vs-loss group -- and
	as a policy improves and its groups tighten, ever-smaller differences are amplified (the
	std-normalization bias Dr. GRPO describes). normalize_std=False keeps advantages in reward
	units: tiny differences give tiny gradients and a penalty coefficient means what it says.
	A group with zero spread is dead either way."""

	idx = defaultdict(list)

	for i, tid in enumerate(task_ids):
		idx[tid].append(i)

	adv = [0.0] * len(rewards)
	for t, ids in idx.items():
		vals = [rewards[i] for i in ids]
		mu = mean(vals)
		sd = pstdev(vals)
		for i in ids:
			if sd < 1e-12:
				adv[i] = 0.0
			else:
				adv[i] = (rewards[i] - mu) / (sd + eps) if normalize_std else rewards[i] - mu

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