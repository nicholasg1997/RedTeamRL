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

	def logprobs_batch(self, pairs: list[tuple[list[int], list[int]]],
	                   use_adapter: bool, with_grad: bool) -> list[torch.Tensor]:
		"""`logprobs` for several (prompt, completion) pairs in ONE forward pass.

		The update scored examples one at a time; at ~1k tokens each that leaves an A100 mostly
		idle between tiny forwards. Rows are LEFT-padded so every sequence ends at the same position:
		the completion tails stay aligned and `logits_to_keep` still materializes only the tail.
		`position_ids` restart at 0 after the padding, so each row sees the same positions it would
		alone, and the attention mask hides the pad. Same math as `logprobs`; the caller bounds the
		batch (padded tokens and logit rows) for memory.
		"""
		if len(pairs) == 1:
			return [self.logprobs(*pairs[0], use_adapter=use_adapter, with_grad=with_grad)]
		device = self.model.device
		pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0
		lengths = [len(p) + len(c) for p, c in pairs]
		width = max(lengths)
		keep = min(max(len(c) for _, c in pairs) + 1, width)
		ids = torch.full((len(pairs), width), pad_id, dtype=torch.long)
		mask = torch.zeros((len(pairs), width), dtype=torch.long)
		for row, ((prompt, completion), length) in enumerate(zip(pairs, lengths)):
			ids[row, width - length:] = torch.tensor(prompt + completion)
			mask[row, width - length:] = 1
		position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
		ids, mask, position_ids = ids.to(device), mask.to(device), position_ids.to(device)
		adapter_ctx = contextlib.nullcontext() if use_adapter else self.model.disable_adapter()
		grad_ctx = contextlib.nullcontext() if with_grad else torch.no_grad()
		with adapter_ctx, grad_ctx:
			logits = self.model(ids, attention_mask=mask, position_ids=position_ids,
			                    logits_to_keep=keep).logits
		out = []
		for row, (_, completion) in enumerate(pairs):
			n_comp = len(completion)
			# Same slice as `logprobs`: completion token i is predicted from the position before it.
			rows = logits[row, keep - n_comp - 1: keep - 1]
			logp = torch.log_softmax(rows, dim=-1)
			targets = torch.tensor(completion, device=device)
			out.append(logp.gather(1, targets.unsqueeze(1)).squeeze(1))
		return out


def prepare_for_long_context_training(model):
	"""Make a LoRA model survive a backward pass over a long sequence. Returns `model`.

	Qwen3-8B saves ~1.14 GiB of activations per layer at 15k tokens -- 41.2 GiB across 36 layers,
	for ONE backward pass, on top of 16.4 GiB of weights. That does not fit on an 80GB card shared
	with two vLLM engines at any memory split. Checkpointing recomputes activations during backward
	instead of storing them: ~30% more compute for roughly 10x less memory.

	The defender phase never needed this (its ~3k-token sequences save ~8.2 GiB), which is why the
	same training code ran fine for a full phase before the attacker's transcripts exposed it.

	Three parts, and the middle one is the trap:
	  1. enable checkpointing;
	  2. `enable_input_require_grads` -- the base model is FROZEN, so a checkpointed segment's
	     inputs do not require grad, its recomputed graph is detached, and backward produces NO
	     LoRA gradient. The run looks healthy and learns nothing;
	  3. disable the KV cache, which checkpointing cannot use.
	"""
	model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
	enable_input_grads = getattr(model, "enable_input_require_grads", None)
	if enable_input_grads is not None:
		enable_input_grads()
	config = getattr(model, "config", None)
	if config is not None:
		config.use_cache = False
	return model
