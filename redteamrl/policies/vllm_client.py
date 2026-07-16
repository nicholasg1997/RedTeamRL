import requests

def make_vllm_generate(
		base_url: str,
		model: str,
		temperature: float = 0.7,
		top_p: float = 0.8,
		max_tokens: int = 512,
		timeout: float = 300.0,
):

	url = f"{base_url.rstrip('/')}/v1/chat/completions"

	def generate(system: str, message: list[dict]) -> str:
		payload = {
			"model": model,
			"messages": [{"role": "system", "content": system}, *message],
			"temperature": temperature,
			"top_p": top_p,
			"max_tokens": max_tokens,
		}

		resp = requests.post(url, json=payload, timeout=timeout)
		resp.raise_for_status()
		return resp.json()["choices"][0]["message"]["content"]

	return generate