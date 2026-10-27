"""An OpenRouter-backed `generate(system, messages) -> str` closure.

Used only to drive a TEACHER model (e.g. Nemotron) as the attacker for the Gate-0 viability probe and,
later, warm-start data collection. OpenRouter is OpenAI-compatible, so this mirrors
`make_vllm_generate` but adds the bearer auth, points at OpenRouter, and drops any separate reasoning
field (the attacker consumes the ANSWER content only). The api key is never echoed into an error.
"""
import time

import requests

_URL = "https://openrouter.ai/api/v1/chat/completions"


def make_openrouter_generate(model, api_key, *, session=None, temperature: float = 0.7,
                             top_p: float = 0.95, max_tokens: int = 8192, timeout: float = 300.0,
                             max_retries: int = 3, backoff_s: float = 2.0,
                             on_response=None):
    """Report every HTTP attempt; missing telemetry remains None, never an invented zero.

    The callback receives metadata only, not prompts, credentials or private reasoning.
    Use a separate closure/session/callback per episode when collecting concurrently.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    http = session or requests
    call_id = 0

    def generate(system: str, messages: list[dict]) -> str:
        nonlocal call_id
        call_id += 1
        payload = {"model": model, "temperature": temperature, "top_p": top_p,
                   "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, *messages]}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        last_err = ""
        for attempt in range(max_retries):
            started = time.monotonic()
            event = {"label": "teacher", "call_id": call_id, "attempt": attempt + 1,
                     "requested_model": model, "max_tokens": max_tokens}

            def report(**values):
                if on_response is not None:
                    on_response({**event, "latency_s": time.monotonic() - started, **values})

            try:
                resp = http.post(_URL, json=payload, headers=headers, timeout=timeout)
            except Exception as exc:
                report(error_type=type(exc).__name__)
                raise RuntimeError(f"OpenRouter transport error: {type(exc).__name__}") from None
            event["http_status"] = resp.status_code
            if resp.status_code == 429:   # rate limited on a free tier -> back off and retry
                last_err = f"429 {resp.text.replace(api_key, '[API_KEY]')[:200]}"
                report(error_type="RateLimit")
                if attempt + 1 < max_retries:
                    time.sleep(backoff_s * (attempt + 1))
                continue
            if resp.status_code != 200:
                report(error_type="HTTPError")
                raise RuntimeError(f"OpenRouter {resp.status_code} for {model}: "
                                   f"{resp.text.replace(api_key, '[API_KEY]')[:400]}")
            try:
                body = resp.json()
                choices = body.get("choices") or []
                choice = choices[0] if choices else {}
                message = choice.get("message") or {}
                usage = body.get("usage") or {}
                content = message.get("content")
                reasoning = message.get("reasoning") or message.get("reasoning_details")
                error = body.get("error") or choice.get("error")
                report(response_id=body.get("id"), model=body.get("model"),
                       provider=body.get("provider"), finish_reason=choice.get("finish_reason"),
                       native_finish_reason=choice.get("native_finish_reason"),
                       usage=usage, prompt_tokens=usage.get("prompt_tokens"),
                       completion_tokens=usage.get("completion_tokens"),
                       reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       cost=usage.get("cost"), final_present=bool(isinstance(content, str) and content.strip()),
                       content_type=type(content).__name__, reasoning_present=bool(reasoning),
                       refusal_present=bool(message.get("refusal")),
                       choices_count=len(choices), content_chars=len(content) if isinstance(content, str) else 0,
                       error_type="ProviderError" if error else "MissingChoices" if not choices else None)
            except (ValueError, TypeError, AttributeError, IndexError):
                report(error_type="ResponseSchemaError")
                raise RuntimeError("OpenRouter response was not valid completion JSON") from None
            if error or not choices:
                raise RuntimeError("OpenRouter returned a provider error; see generation metadata")
            # A reasoning model can spend its whole budget thinking and emit no answer. Degrade that
            # turn to "" so the agent consumes it (fail-closed) and the episode continues, rather than
            # crashing the whole rollout on one odd response.
            return content if isinstance(content, str) else ""
        raise RuntimeError(f"OpenRouter {model} rate-limited after {max_retries} retries: {last_err}")

    return generate
