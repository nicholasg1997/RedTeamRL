import subprocess, time, requests


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


def start_vllm_server(model: str, port: int, mem_frac: float, max_model_len: int = 8192,
                      max_num_seqs: int | None = None, timeout_s: int = 1200, poll_s: int = 2) -> None:
	"""Launch `vllm serve <model>` and block until its /health is 200. Shared by the eval probe
	and the training harness. `max_num_seqs` caps concurrent sequences — needed for hybrid
	(Mamba) models at low gpu_memory_utilization, where vLLM's default 256 can exceed the cache
	blocks that fit. Raises if it never becomes healthy within timeout_s."""
	cmd = ["vllm", "serve", model, "--port", str(port),
	       "--gpu-memory-utilization", str(mem_frac), "--max-model-len", str(max_model_len)]
	if max_num_seqs is not None:
		cmd += ["--max-num-seqs", str(max_num_seqs)]
	subprocess.Popen(cmd)
	for _ in range(max(1, timeout_s // poll_s)):
		try:
			if requests.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
				return
		except Exception:
			pass
		time.sleep(poll_s)
	raise RuntimeError(f"vLLM for {model} on :{port} never became healthy in {timeout_s}s")
