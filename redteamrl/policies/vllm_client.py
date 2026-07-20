import requests


def _prompt_stats(system: str, messages: list[dict]) -> str:
	message_chars = [len(str(message.get("content", ""))) for message in messages]
	roles = ",".join(str(message.get("role", "?")) for message in messages)
	return (
		f"system_chars={len(system)} "
		f"message_chars={message_chars} "
		f"total_chars={len(system) + sum(message_chars)} "
		f"roles={roles}"
	)


def make_vllm_generate(
		base_url: str,
		model: str,
		temperature: float = 0.7,
		top_p: float = 0.8,
		max_tokens: int = 512,
		timeout: float = 300.0,
		enable_thinking: bool = False,
		label: str = "",
		log_prompt_stats: bool = False,
):

	url = f"{base_url.rstrip('/')}/v1/chat/completions"

	def generate(system: str, messages: list[dict]) -> str:
		stats = _prompt_stats(system, messages)
		if log_prompt_stats:
			prefix = f"[{label}] " if label else ""
			print(f"{prefix}vLLM prompt stats: {stats}")

		payload = {
			"model": model,
			"messages": [{"role": "system", "content": system}, *messages],
			"temperature": temperature,
			"top_p": top_p,
			"max_tokens": max_tokens,
			"chat_template_kwargs": {"enable_thinking": enable_thinking},
		}

		resp = requests.post(url, json=payload, timeout=timeout)
		if resp.status_code != 200:
			# Surface vLLM's error body — raise_for_status() drops it, and that body
			# (context-length overflow vs OOM vs bad field) is the whole diagnosis.
			prefix = f" [{label}]" if label else ""
			raise RuntimeError(f"vLLM{prefix} {resp.status_code} from {url} ({stats}): {resp.text[:800]}")
		return resp.json()["choices"][0]["message"]["content"]

	return generate
