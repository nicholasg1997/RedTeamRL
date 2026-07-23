"""Non-collapse metrics for the training loop (adopted from MORPHEUS's evaluation protocol).

Until now defender collapse was caught by eyeballing reward histograms. These three metrics make it
measurable. All are pure — no model, no Modal — so they are unit-tested offline; the caller supplies
the activations / reward histories.

  effective_rank  plasticity. Declining rank WITHOUT performance recovery = plasticity collapse.
  forgetting      per-task regression from its own peak — cycling, in a self-play setting.
  stability       within-task reward variance; read it WITH the mean (see docstring).

Adaptation speed and recovery time from the same protocol need a defined regime shift, so they only
apply once the opponent population starts shifting — not against a frozen opponent.
See docs/ROADMAP.md ("Measuring non-collapse") and the reference-morpheus memory.
"""
import torch


def effective_rank(activations, eps: float = 1e-12) -> float:
	"""Entropy-based effective rank (Roy & Vetterli) of a 2D samples x features matrix.

	exp(H(p)) where p is the singular-value spectrum normalised to a distribution. A rank-1 matrix
	scores 1.0; k equally-weighted directions score k. Track it on a FIXED probe batch across
	iterations — only then are successive values comparable, and a downward drift means the policy's
	representation is collapsing onto fewer directions (loss of plasticity, Dohare et al.)."""
	matrix = torch.as_tensor(activations, dtype=torch.float32)
	if matrix.ndim != 2:
		raise ValueError(f"effective_rank expects a 2D matrix (samples x features), got {matrix.ndim}D")
	singular = torch.linalg.svdvals(matrix)
	singular = singular[singular > eps]
	if singular.numel() == 0:
		return 0.0
	p = singular / singular.sum()
	return float(torch.exp(-(p * p.log()).sum()))


def stability(task_rewards: dict[str, list[float]]) -> dict[str, dict[str, float]]:
	"""Per-task reward mean and (population) variance.

	Variance alone is ambiguous — read it with the mean: low variance + LOW mean is a converged but
	suboptimal policy (a saturated/dead group), low variance + HIGH mean is genuine competence.
	Tasks with no rollouts are omitted rather than reported as zero."""
	out: dict[str, dict[str, float]] = {}
	for task_id, rewards in task_rewards.items():
		if not rewards:
			continue
		mean = sum(rewards) / len(rewards)
		var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
		out[task_id] = {"mean": mean, "var": var}
	return out


def forgetting(history: list[dict[str, float]]) -> dict[str, float]:
	"""Per-task peak-so-far minus current mean reward, over a list of per-iteration {task_id: mean}.

	POSITIVE = the model regressed on a task it previously handled better (backward degradation); in
	self-play that is the direct signature of cycling — relearning one attack while losing another.
	Negative means it is still improving. Tasks with no prior history are skipped, and fewer than two
	iterations yields nothing."""
	if len(history) < 2:
		return {}
	current = history[-1]
	out: dict[str, float] = {}
	for task_id, now in current.items():
		prior = [past[task_id] for past in history[:-1] if task_id in past]
		if prior:
			out[task_id] = max(prior) - now
	return out
