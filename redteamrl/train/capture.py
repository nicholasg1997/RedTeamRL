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


class VLLMCapturingGenerate:
	"""`generate(system, messages) -> str` served by vLLM, capturing the EXACT sampled token ids.

	Drop-in for CapturingGenerate on the generation side. HF `model.generate()` at batch-1 was
	~70% of GRPO wall clock: it holds the GIL, so concurrent episodes serialize instead of
	overlapping. vLLM batches them properly.

	The ids come from vLLM's `return_token_ids`, never from retokenizing the returned text. The
	same string can tokenize differently than it was generated, and GRPO would then score a
	sequence the policy never sampled — a wrong gradient with no visible symptom.

	The prompt is rendered with the LOCAL chat template and sent as text with
	`add_special_tokens=False`, so the ids vLLM reports are the ids HF re-scores during the update.
	"""

	def __init__(self, base_url: str, model: str, tokenizer, temperature: float = 0.7,
	             max_new_tokens: int = 512, enable_thinking: bool = False, top_p: float = 0.8,
	             timeout: float = 600.0, post=None):
		self.url = f"{base_url.rstrip('/')}/v1/completions"
		self.model = model
		self.tokenizer = tokenizer
		self.temperature = temperature
		self.max_new_tokens = max_new_tokens
		self.enable_thinking = enable_thinking
		self.top_p = top_p
		self.timeout = timeout
		self._post = post
		self.buffer: list[Example] = []
		self._local = threading.local()

	@contextlib.contextmanager
	def episode_capture(self):
		"""Per-episode sink; see CapturingGenerate.episode_capture."""
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
		from redteamrl.policies.vllm_client import parse_token_ids

		chat = [{"role": "system", "content": system}, *messages]
		prompt = self.tokenizer.apply_chat_template(
			chat, add_generation_prompt=True, enable_thinking=self.enable_thinking, tokenize=False
		)
		post = self._post
		if post is None:
			import requests
			post = requests.post
		response = post(
			self.url,
			json={
				"model": self.model,
				"prompt": prompt,
				"max_tokens": self.max_new_tokens,
				"temperature": self.temperature,
				"top_p": self.top_p,
				# The chat template already carries BOS; a second one would shift every id
				# relative to what HF re-scores.
				"add_special_tokens": False,
				"return_token_ids": True,
			},
			timeout=self.timeout,
		)
		if response.status_code != 200:
			raise RuntimeError(f"vLLM {response.status_code} from {self.url}: "
			                   f"{getattr(response, 'text', '')[:500]}")
		text, prompt_ids, completion_ids = parse_token_ids(response.json())
		self._sink().append(Example(prompt_ids=prompt_ids, completion_ids=completion_ids))
		return text
