import contextlib
import torch

class Learner:
	"""Score completions under the live adapter and its adapter-disabled reference.

	The caller owns the reference construction. For defender GRPO, ``train_defender.py`` first
	merges the accepted SFT bootstrap into the backbone and then attaches a new trainable adapter;
	disabling that adapter therefore returns the frozen SFT policy, not raw Qwen.
	"""

	def __init__(self, model, tokenizer, optimizer, reward_key: str = "defender_reward"):
		self.model = model
		self.tokenizer = tokenizer
		self.optimizer = optimizer
		self.reward_key = reward_key

	def logprobs(self, prompt_ids: list[int], completion_ids: list[int],
	             use_adapter: bool, with_grad: bool) -> torch.Tensor:
		ids = torch.tensor([prompt_ids + completion_ids], device=self.model.device)
		n_prompt = len(prompt_ids)
		n_comp = len(completion_ids)
		adapter_ctx = contextlib.nullcontext() if use_adapter else self.model.disable_adapter()
		grad_ctx = contextlib.nullcontext() if with_grad else torch.no_grad()
		# Only the completion positions are ever used, and Qwen3's ~152k vocab makes the
		# DISCARDED prompt logits the dominant allocation: a 15k-token sequence materializes
		# 4.25 GiB in bf16 to keep 0.58 GiB of it, which is what OOM'd the attacker phase on an
		# 80GB A100. `logits_to_keep` runs the LM head over only the tail we need.
		# One EXTRA position: completion token i is predicted from position n_prompt-1+i, so the
		# first completion token's logits live at the prompt's last position.
		keep = min(n_comp + 1, ids.shape[1])
		with adapter_ctx, grad_ctx:
			logits = self.model(ids, attention_mask=torch.ones_like(ids),
			                    logits_to_keep=keep).logits[0]
		logits = logits[:n_comp]
		logp = torch.log_softmax(logits, dim=-1)
		targets = torch.tensor(completion_ids, device=self.model.device)
		return logp.gather(1, targets.unsqueeze(1)).squeeze(1)
