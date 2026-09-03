import json

import torch
import torch.nn.functional as F


def _target_segments(target: str, decision_loss_weight: float) -> list[tuple[str, float]]:
	"""Split canonical defender JSON into reasoning and decision-weighted suffixes."""
	try:
		decision = json.loads(target)
	except (TypeError, json.JSONDecodeError):
		return [(target, 1.0)]
	if not isinstance(decision, dict) or "reasoning" not in decision or "verdict" not in decision:
		return [(target, 1.0)]

	reasoning_segment = json.dumps({"reasoning": decision["reasoning"]})[:-1]
	decision_fields = {
		key: value for key, value in decision.items()
		if key != "reasoning"
	}
	decision_segment = ", " + json.dumps(decision_fields)[1:]
	if reasoning_segment + decision_segment != target:
		# Only canonical targets created by render_target receive field-aware weighting. A legacy or
		# externally-authored target still trains normally instead of failing the whole run.
		return [(target, 1.0)]
	return [(reasoning_segment, 1.0), (decision_segment, decision_loss_weight)]


def _last_json_span(raw: str) -> tuple[int, int] | None:
	"""Return the (start, end) char span of the LAST top-level JSON object in `raw`, or None.

	Mirrors the scan in `redteamrl.harness.protocol._extract_json`, but returns the object's
	position instead of the parsed object itself, so a caller can split `raw` into a reasoning
	prefix and an action suffix. On each decode we jump the cursor past the consumed span, so a
	nested object (like an `args` dict) is never mistaken for a top-level one.
	"""
	decoder = json.JSONDecoder()
	found = None
	i, n = 0, len(raw)
	while i < n:
		if raw[i] == "{":
			try:
				obj, end = decoder.raw_decode(raw[i:])
			except json.JSONDecodeError:
				i += 1
				continue
			if isinstance(obj, dict):
				found = (i, i + end)
				i += end
				continue
		i += 1
	return found


def _attacker_target_segments(target: str, action_loss_weight: float) -> list[tuple[str, float]]:
	"""Split an attacker target `reasoning\n{json action}` into a reasoning prefix trained at
	weight 1.0 and the trailing JSON-action span trained at `action_loss_weight` (0.0 masks a
	rejected/undesirable action out of the gradient entirely). Falls back to training the whole
	target at 1.0 if no JSON object can be found."""
	span = _last_json_span(target)
	if span is None:
		return [(target, 1.0)]
	start, stop = span
	return [(target[:start], 1.0), (target[start:stop], action_loss_weight)]


def encode_example(
	tok,
	ex: dict,
	decision_loss_weight: float = 5.0,
) -> tuple[list[int], list[int], list[float]]:
	if decision_loss_weight <= 0:
		raise ValueError("decision_loss_weight must be positive")
	chat = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["prompt"]}]
	prompt_ids = tok.apply_chat_template(
		chat, add_generation_prompt=True, enable_thinking=False
	)
	target_ids: list[int] = []
	target_weights: list[float] = []
	is_attacker_example = "action_loss_weight" in ex
	if is_attacker_example:
		segments = _attacker_target_segments(ex["target"], ex["action_loss_weight"])
	else:
		segments = _target_segments(ex["target"], decision_loss_weight)
	for segment, weight in segments:
		segment_ids = tok(segment, add_special_tokens=False)["input_ids"]
		target_ids.extend(segment_ids)
		target_weights.extend([weight] * len(segment_ids))
	target_ids.append(tok.eos_token_id)
	# A masked action masks its terminator too, so the eos weight always tracks the action weight
	# for an attacker example — including the (rare) fallback where no JSON span was found.
	target_weights.append(ex["action_loss_weight"] if is_attacker_example else segments[-1][1])
	input_ids = prompt_ids + target_ids
	labels = [-100] * len(prompt_ids) + target_ids
	loss_weights = [0.0] * len(prompt_ids) + target_weights
	return input_ids, labels, loss_weights

def _pad_microbatch(tok, encoded, device):
	pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
	max_len = max(len(input_ids) for input_ids, _, _ in encoded)
	input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=device)
	labels = torch.full((len(encoded), max_len), -100, dtype=torch.long, device=device)
	attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
	loss_weights = torch.zeros((len(encoded), max_len), dtype=torch.float32, device=device)
	for row, (example_ids, example_labels, example_weights) in enumerate(encoded):
		length = len(example_ids)
		input_ids[row, :length] = torch.tensor(example_ids, dtype=torch.long, device=device)
		labels[row, :length] = torch.tensor(example_labels, dtype=torch.long, device=device)
		attention_mask[row, :length] = 1
		loss_weights[row, :length] = torch.tensor(
			example_weights, dtype=torch.float32, device=device
		)
	return input_ids, labels, attention_mask, loss_weights


def _weighted_causal_loss(outputs, labels, loss_weights):
	logits = getattr(outputs, "logits", None)
	if logits is None:
		# Lightweight test doubles and unusual model wrappers may only expose their own scalar loss.
		return outputs.loss
	shift_logits = logits[:, :-1, :].contiguous()
	shift_labels = labels[:, 1:].contiguous()
	shift_weights = loss_weights[:, 1:].contiguous()
	token_losses = F.cross_entropy(
		shift_logits.view(-1, shift_logits.size(-1)),
		shift_labels.view(-1),
		ignore_index=-100,
		reduction="none",
	).view_as(shift_labels)
	valid_weights = shift_weights * (shift_labels != -100)
	denominator = valid_weights.sum()
	if denominator.item() <= 0:
		raise ValueError("SFT microbatch contains no supervised target tokens")
	return (token_losses * valid_weights).sum() / denominator


def _encoded_weight(encoded_example) -> float:
	_, labels, loss_weights = encoded_example
	return sum(
		weight for label, weight in zip(labels, loss_weights)
		if label != -100
	)


def sft_loss(
	model,
	tok,
	examples: list[dict],
	microbatch_size: int = 2,
	decision_loss_weight: float = 5.0,
) -> float:
	"""Teacher-forced target-token loss without updating weights."""
	if not examples:
		return 0.0
	if microbatch_size <= 0:
		raise ValueError("microbatch_size must be positive")
	encoded = [
		encode_example(tok, ex, decision_loss_weight=decision_loss_weight)
		for ex in examples
	]
	encoded.sort(key=lambda pair: len(pair[0]))
	total_loss_weight = sum(_encoded_weight(example) for example in encoded)
	total_loss = 0.0
	was_training = model.training
	model.eval()
	try:
		with torch.inference_mode():
			for start in range(0, len(encoded), microbatch_size):
				microbatch = encoded[start:start + microbatch_size]
				ii, ll, attention_mask, loss_weights = _pad_microbatch(
					tok, microbatch, model.device
				)
				microbatch_loss_weight = sum(_encoded_weight(example) for example in microbatch)
				outputs = model(input_ids=ii, attention_mask=attention_mask, labels=ll)
				loss = _weighted_causal_loss(outputs, ll, loss_weights)
				if not torch.isfinite(loss):
					raise FloatingPointError("non-finite SFT evaluation loss")
				total_loss += loss.item() * microbatch_loss_weight
	finally:
		model.train(was_training)
	return total_loss / total_loss_weight


def sft_step(
	model,
	tok,
	batch: list[dict],
	opt,
	max_grad_norm: float = 1.0,
	microbatch_size: int = 2,
	decision_loss_weight: float = 5.0,
) -> dict:
	if not batch:
		return {"loss": 0.0, "n": 0}
	if microbatch_size <= 0:
		raise ValueError("microbatch_size must be positive")
	opt.zero_grad()
	encoded = [
		encode_example(tok, ex, decision_loss_weight=decision_loss_weight)
		for ex in batch
	]
	encoded.sort(key=lambda pair: len(pair[0]))  # bucket similar lengths to reduce padding
	total_loss_weight = sum(_encoded_weight(example) for example in encoded)
	total_loss = 0.0

	for start in range(0, len(encoded), microbatch_size):
		microbatch = encoded[start:start + microbatch_size]
		ii, ll, attention_mask, loss_weights = _pad_microbatch(
			tok, microbatch, model.device
		)
		microbatch_loss_weight = sum(_encoded_weight(example) for example in microbatch)
		outputs = model(input_ids=ii, attention_mask=attention_mask, labels=ll)
		loss = _weighted_causal_loss(outputs, ll, loss_weights)
		if not torch.isfinite(loss):
			raise FloatingPointError("non-finite SFT training loss")
		# Accumulate a single decision-weighted mean over the logical batch, independent of how it
		# is divided into padded microbatches.
		weight = microbatch_loss_weight / total_loss_weight
		(loss * weight).backward()
		total_loss += loss.item() * microbatch_loss_weight
	if max_grad_norm > 0:
		grad_norm = torch.nn.utils.clip_grad_norm_(
			[p for p in model.parameters() if p.requires_grad], max_grad_norm
		)
		if not torch.isfinite(grad_norm):
			raise FloatingPointError("non-finite SFT gradient norm")
	opt.step()
	return {"loss": total_loss / total_loss_weight, "n": len(batch)}
