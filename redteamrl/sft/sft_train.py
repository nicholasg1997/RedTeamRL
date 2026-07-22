import torch

def encode_example(tok, ex: dict) -> tuple[list[int], list[int]]:
	chat = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["prompt"]}]
	prompt_ids = tok.apply_chat_template(chat, add_generation_prompt=True, enable_thinking=False) # maybe turn thinking on
	target_ids = tok(ex['target'], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
	input_ids = prompt_ids + target_ids
	labels = [-100] * len(prompt_ids) + target_ids
	return input_ids, labels

def sft_step(model, tok, batch: list[dict], opt) -> dict:
	opt.zero_grad()
	total = 0.0
	n = 0

	for ex in batch:
		input_ids, labels = encode_example(tok, ex)
		ii = torch.tensor([input_ids], device=model.device)
		ll = torch.tensor([labels], device=model.device)
		loss = model(input_ids=ii, attention_mask=torch.ones_like(ii), labels=ll).loss
		loss.backward()
		total += loss.item()
		n += 1
	opt.step()
	return {"loss": total / max(n, 1), "n": n}