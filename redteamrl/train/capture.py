import contextlib
import threading
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
	verdicts: list[dict[str, str]] = field(default_factory=list)
	episode_type: str = ""
	policy_leak: bool = False
	complete: bool = False
	defender_protocol_failures: int = 0


class CapturingGenerate:
	def __init__(self, model, tokenizer, temperature: float = 0.7, max_new_tokens: int = 512,
	             enable_thinking: bool = False):
		self.model = model
		self.tokenizer = tokenizer
		self.temperature = temperature
		self.max_new_tokens = max_new_tokens
		self.enable_thinking = enable_thinking          # OFF by default: a monitor emits reasoning+JSON, not a think block
		self.buffer: list[Example] = []                 # fallback sink when no episode is scoped
		self._local = threading.local()

	@contextlib.contextmanager
	def episode_capture(self):
		"""Route this thread's captures into a private buffer for the duration of one episode.

		Concurrent episodes share one CapturingGenerate. Slicing a single shared buffer by index
		interleaves examples across episodes and tags them with another episode's reward -- silent
		credit-assignment corruption. Giving each episode its own sink is what makes concurrency
		safe, and concurrency is what lets vLLM batch the frozen opponent instead of serving it
		one request at a time.
		"""
		episode_buffer: list[Example] = []
		previous = getattr(self._local, "buffer", None)
		self._local.buffer = episode_buffer
		try:
			yield episode_buffer
		finally:
			self._local.buffer = previous

	def _sink(self) -> list[Example]:
		scoped = getattr(self._local, "buffer", None)
		return self.buffer if scoped is None else scoped

	def __call__(self, system: str, messages: list[dict]) -> str:
		chat = [{"role": "system", "content": system}, *messages]
		prompt_ids = self.tokenizer.apply_chat_template(
			chat, add_generation_prompt=True, enable_thinking=self.enable_thinking, return_tensors="pt"
		).to(self.model.device)

		with torch.no_grad():
			full = self.model.generate(
				prompt_ids, attention_mask=torch.ones_like(prompt_ids),
				do_sample = self.temperature > 0, temperature=self.temperature,
				max_new_tokens=self.max_new_tokens
			)
		n = prompt_ids.shape[1]
		completions_ids = full[0, n:].tolist()
		self._sink().append(Example(
			prompt_ids=prompt_ids[0].tolist(),
			completion_ids=completions_ids,
		))
		return self.tokenizer.decode(completions_ids, skip_special_tokens=True)
