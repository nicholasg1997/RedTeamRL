import torch

def encode_example(tok, ex: dict) -> tuple[list[int], list[int]]:
	chat = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["prompt"]}]
	prompt_ids = tok.apply_chat_template(chat, add_generation_prompt=True, enable_thinking=False) # maybe turn thinking on
	target_ids = tok(ex['target'], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
	input_ids = prompt_ids + target_ids
	labels = [-100] * len(prompt_ids) + target_ids
	return input_ids, labels

def _pad_microbatch(tok, encoded, device):
	pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
	max_len = max(len(input_ids) for input_ids, _ in encoded)
	input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=device)
	labels = torch.full((len(encoded), max_len), -100, dtype=torch.long, device=device)
	attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
	for row, (example_ids, example_labels) in enumerate(encoded):
		length = len(example_ids)
		input_ids[row, :length] = torch.tensor(example_ids, dtype=torch.long, device=device)
		labels[row, :length] = torch.tensor(example_labels, dtype=torch.long, device=device)
		attention_mask[row, :length] = 1
	return input_ids, labels, attention_mask


def sft_loss(model, tok, examples: list[dict], microbatch_size: int = 2) -> float:
	"""Teacher-forced target-token loss without updating weights."""
	if not examples:
		return 0.0
	if microbatch_size <= 0:
		raise ValueError("microbatch_size must be positive")
	encoded = [encode_example(tok, ex) for ex in examples]
	encoded.sort(key=lambda pair: len(pair[0]))
	total_target_tokens = sum(sum(label != -100 for label in labels) for _, labels in encoded)
	total_loss = 0.0
	was_training = model.training
	model.eval()
	try:
		with torch.inference_mode():
			for start in range(0, len(encoded), microbatch_size):
				microbatch = encoded[start:start + microbatch_size]
				ii, ll, attention_mask = _pad_microbatch(tok, microbatch, model.device)
				microbatch_target_tokens = int((ll != -100).sum().item())
				loss = model(input_ids=ii, attention_mask=attention_mask, labels=ll).loss
				if not torch.isfinite(loss):
					raise FloatingPointError("non-finite SFT evaluation loss")
				total_loss += loss.item() * microbatch_target_tokens
	finally:
		model.train(was_training)
	return total_loss / total_target_tokens


def sft_step(
	model,
	tok,
	batch: list[dict],
	opt,
	max_grad_norm: float = 1.0,
	microbatch_size: int = 2,
) -> dict:
	if not batch:
		return {"loss": 0.0, "n": 0}
	if microbatch_size <= 0:
		raise ValueError("microbatch_size must be positive")
	opt.zero_grad()
	encoded = [encode_example(tok, ex) for ex in batch]
	encoded.sort(key=lambda pair: len(pair[0]))  # bucket similar lengths to reduce padding
	total_target_tokens = sum(sum(label != -100 for label in labels) for _, labels in encoded)
	total_loss = 0.0

	for start in range(0, len(encoded), microbatch_size):
		microbatch = encoded[start:start + microbatch_size]
		ii, ll, attention_mask = _pad_microbatch(tok, microbatch, model.device)
		microbatch_target_tokens = int((ll != -100).sum().item())
		loss = model(input_ids=ii, attention_mask=attention_mask, labels=ll).loss
		if not torch.isfinite(loss):
			raise FloatingPointError("non-finite SFT training loss")
		# Hugging Face returns a mean over supervised tokens. Weight each microbatch by its token
		# count so accumulated gradients equal one token-mean loss over the full logical batch.
		weight = microbatch_target_tokens / total_target_tokens
		(loss * weight).backward()
		total_loss += loss.item() * microbatch_target_tokens
	if max_grad_norm > 0:
		grad_norm = torch.nn.utils.clip_grad_norm_(
			[p for p in model.parameters() if p.requires_grad], max_grad_norm
		)
		if not torch.isfinite(grad_norm):
			raise FloatingPointError("non-finite SFT gradient norm")
	opt.step()
	return {"loss": total_loss / total_target_tokens, "n": len(batch)}
