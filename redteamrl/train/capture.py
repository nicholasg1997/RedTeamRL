from dataclasses import dataclass, field
import torch

@dataclass
class Example:
	prompt_ids: list[int]
	completion_ids: list[int]
	task_id: str = ""
	episode_id: int = -1
	reward: float = 0.0
	advantage: float = 0.0


class CapturingGenerate:
	def __init__(self, model, tokenizer, temperature: float = 0.7, max_new_tokens: int = 512,):
		self.model = model
		self.tokenizer = tokenizer
		self.temperature = temperature
		self.max_new_tokens = max_new_tokens
		self.buffer: list[Example] = []

	def __call__(self, system: str, messages: list[dict]) -> str:
		chat = [{"role": "system", "content": system}, *messages]
		prompt_ids = self.tokenizer.apply_chat_template(
			chat, add_generation_prompt=True, return_tensors="pt"
		).to(self.model.device)

		with torch.no_grad():
			full = self.model.generate(
				prompt_ids, attention_mask=torch.ones_like(prompt_ids),
				do_sample = self.temperature > 0, temperature=self.temperature,
				max_new_tokens=self.max_new_tokens
			)
		n = prompt_ids.shape[1]
		completions_ids = full[0, n:].tolist()
		self.buffer.append(Example(
			prompt_ids=prompt_ids[0].tolist(),
			completion_ids=completions_ids,
		))
		return self.tokenizer.decode(completions_ids, skip_special_tokens=True)