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


def build_serve_command(model: str, port: int, mem_frac: float, max_model_len: int = 8192,
                        max_num_seqs: int | None = None, lora_name: str | None = None,
                        lora_path: str | None = None, max_lora_rank: int = 16) -> list[str]:
	"""Assemble the `vllm serve` argv. Split out from the launcher so it is testable offline."""
	if lora_path and not lora_name:
		raise ValueError("lora_path requires a lora_name to address it in requests")
	cmd = ["vllm", "serve", model, "--port", str(port),
	       "--gpu-memory-utilization", str(mem_frac), "--max-model-len", str(max_model_len)]
	if max_num_seqs is not None:
		cmd += ["--max-num-seqs", str(max_num_seqs)]
	if lora_name:
		# Runtime swapping additionally needs VLLM_ALLOW_RUNTIME_LORA_UPDATING=True in the env.
		cmd += ["--enable-lora", "--max-lora-rank", str(max_lora_rank)]
		if lora_path:
			cmd += ["--lora-modules", f"{lora_name}={lora_path}"]
	return cmd


def parse_token_ids(payload: dict) -> tuple[str, list[int], list[int]]:
	"""Extract (text, prompt_ids, completion_ids) from a `return_token_ids` completion response.

	GRPO recomputes log-probs under HF for the sequence that was SAMPLED. Retokenizing the
	returned text is not equivalent -- the same string can tokenize differently than it was
	generated -- and the resulting importance ratio would be silently wrong. So a response
	without ids is an error, never a fallback.

	Field placement has moved between vLLM versions, so both the choice and the top level are
	accepted; the probe script prints the real schema for this stack.
	"""
	choices = payload.get("choices") or []
	if not choices:
		raise ValueError(f"vLLM response has no choices: {str(payload)[:300]}")
	choice = choices[0]

	def pick(*names):
		for name in names:
			for holder in (choice, payload):
				value = holder.get(name)
				if value is not None:
					return list(value)
		return None

	prompt_ids = pick("prompt_token_ids")
	completion_ids = pick("token_ids", "completion_token_ids", "output_token_ids")
	if prompt_ids is None or completion_ids is None:
		raise ValueError(
			"vLLM response carries no token ids — request return_token_ids=True (retokenizing "
			f"the text would corrupt the importance ratio). keys={sorted(choice)} / "
			f"{sorted(payload)}"
		)
	if not completion_ids:
		raise ValueError("vLLM returned an empty completion; nothing to train on")
	return choice.get("text", ""), prompt_ids, completion_ids


def load_lora_adapter(base_url: str, name: str, path: str, timeout: float = 300.0) -> None:
	"""Swap the served adapter in place. This is how the GRPO policy reaches vLLM each iteration:
	a ~50MB adapter file instead of streaming the whole backbone."""
	resp = requests.post(
		f"{base_url.rstrip('/')}/v1/load_lora_adapter",
		json={"lora_name": name, "lora_path": path, "load_inplace": True},
		timeout=timeout,
	)
	if resp.status_code != 200:
		raise RuntimeError(
			f"vLLM refused the adapter swap ({resp.status_code}): {resp.text[:500]}. "
			"Serving a stale adapter would train against a policy that is not the one being "
			"updated, so this must not be ignored."
		)


def start_vllm_server(model: str, port: int, mem_frac: float, max_model_len: int = 8192,
                      max_num_seqs: int | None = None, timeout_s: int = 1200, poll_s: int = 2,
                      lora_name: str | None = None, lora_path: str | None = None,
                      max_lora_rank: int = 16) -> subprocess.Popen:
	"""Launch `vllm serve <model>` and block until its /health is 200. Shared by the eval probe
	and the training harness. `max_num_seqs` caps concurrent sequences — needed for hybrid
	(Mamba) models at low gpu_memory_utilization, where vLLM's default 256 can exceed the cache
	blocks that fit. Raises if it never becomes healthy within timeout_s."""
	cmd = build_serve_command(model, port, mem_frac, max_model_len, max_num_seqs,
	                          lora_name, lora_path, max_lora_rank)
	proc = subprocess.Popen(cmd)
	for _ in range(max(1, timeout_s // poll_s)):
		try:
			if requests.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
				return proc
		except Exception:
			pass
		time.sleep(poll_s)
	stop_vllm_server(proc)
	raise RuntimeError(f"vLLM for {model} on :{port} never became healthy in {timeout_s}s")


def stop_vllm_server(proc: subprocess.Popen | None, timeout_s: int = 60) -> None:
	"""Release a vLLM process and its GPU allocation, with a bounded graceful shutdown."""
	if proc is None or proc.poll() is not None:
		return
	proc.terminate()
	try:
		proc.wait(timeout=timeout_s)
	except subprocess.TimeoutExpired:
		proc.kill()
		proc.wait()
