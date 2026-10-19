"""Verify a runtime LoRA swap by comparing vLLM to ITSELF, never to HF.

The old publish check compared vLLM greedy text to HF greedy text and raised "serving the BASE
policy" when vLLM matched HF-base but not HF+adapter. After a small update the served adapter is
~base, so one near-tied token flipped differently by the two stacks ("I'm" vs "I am") tripped it on a
swap that had worked (2026-10-18, v3 arm 2 iter 0, ratio=1.000). Two engines cannot be the oracle.

Here the oracle is the same engine: load the SAME adapter file under a never-used name (no cache
can serve a stale copy of a name it has never seen) and require the served name's first-token
log-prob distribution to match it. A swap that did not take serves the PREVIOUS adapter, which
diverges from the fresh load -- more each iteration as training moves the policy.
"""
from __future__ import annotations

import itertools

import requests

_counter = itertools.count()


def dist_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Mean |logprob difference| over the union of two top-K maps; a token missing from one map is
    floored just below that map's K-th logprob."""
    floor_p, floor_q = min(p.values()) - 1.0, min(q.values()) - 1.0
    keys = set(p) | set(q)
    return sum(abs(p.get(k, floor_p) - q.get(k, floor_q)) for k in keys) / len(keys)


def signature_distance(a: list[dict], b: list[dict]) -> float:
    return sum(dist_distance(x, y) for x, y in zip(a, b)) / len(a)


def first_token_signature(base_url: str, model: str, prompts: list[str], top_k: int = 20,
                          timeout: float = 120.0) -> list[dict]:
    """Top-K log-probs of the first generated token for each prompt (greedy, sequential)."""
    sig = []
    for prompt in prompts:
        r = requests.post(f"{base_url.rstrip('/')}/v1/completions",
                          json={"model": model, "prompt": prompt, "max_tokens": 1,
                                "temperature": 0.0, "logprobs": top_k,
                                "add_special_tokens": False}, timeout=timeout)
        r.raise_for_status()
        sig.append(r.json()["choices"][0]["logprobs"]["top_logprobs"][0])
    return sig


class SwapMismatch(RuntimeError):
    """The served name does not match a fresh load. `.measurement` holds every distance, so the
    caller can log the evidence and decide whether a reload is worth one more try."""

    def __init__(self, message: str, measurement: dict):
        super().__init__(message)
        self.measurement = measurement


def _top1_agreement(a: list[dict], b: list[dict]) -> float:
    return sum(max(x, key=x.get) == max(y, key=y.get) for x, y in zip(a, b)) / len(a)


def measure_served_adapter(base_url: str, served_name: str, adapter_path: str, base_model: str,
                           prompts: list[str], previous_path: str | None = None) -> dict:
    """Every distance needed to tell a STALE swap from a NOISY check.

    v4 arm 2 iter 1 (2026-10-18) raised at 0.0853 right after a successful prefix-cache reset, and
    the single number could not say which it was. So, besides served-vs-fresh:
      served_repeat      -- the served name measured twice. Nonzero means the engine is not
                            bit-deterministic here and the tolerance is measuring noise.
      served_vs_previous -- the served name vs a fresh load of the PREVIOUS adapter. ~0 means the
                            swap really did not take (vLLM still serves last iteration's weights).
      top1_agree         -- served vs fresh greedy-token agreement; a stale swap after a tiny update
                            can still agree, a tail-only wobble always does.
    """
    base_url = base_url.rstrip("/")
    fresh = f"swapcheck-{next(_counter)}"
    prev = f"swapcheck-{next(_counter)}" if previous_path else None
    loaded = []
    try:
        for name, path in ((fresh, adapter_path), (prev, previous_path)):
            if name is None:
                continue
            r = requests.post(f"{base_url}/v1/load_lora_adapter",
                              json={"lora_name": name, "lora_path": path}, timeout=300)
            r.raise_for_status()
            loaded.append(name)
        served_sig = first_token_signature(base_url, served_name, prompts)
        fresh_sig = first_token_signature(base_url, fresh, prompts)
        served_again = first_token_signature(base_url, served_name, prompts)
        base_sig = first_token_signature(base_url, base_model, prompts)
        prev_sig = first_token_signature(base_url, prev, prompts) if prev else None
    finally:
        for name in loaded:
            requests.post(f"{base_url}/v1/unload_lora_adapter", json={"lora_name": name},
                          timeout=60)
    out = {"served_vs_fresh": signature_distance(served_sig, fresh_sig),
           "served_repeat": signature_distance(served_sig, served_again),
           "fresh_vs_base": signature_distance(fresh_sig, base_sig),
           "top1_agree": _top1_agreement(served_sig, fresh_sig)}
    if prev_sig is not None:
        out["served_vs_previous"] = signature_distance(served_sig, prev_sig)
    return out


def describe(m: dict) -> str:
    parts = [f"served vs fresh load {m['served_vs_fresh']:.4f}",
             f"repeat {m['served_repeat']:.4f}", f"top1 agree {m['top1_agree']:.0%}"]
    if "served_vs_previous" in m:
        parts.append(f"vs previous adapter {m['served_vs_previous']:.4f}")
    parts.append(f"adapter moved {m['fresh_vs_base']:.4f} from base")
    return "; ".join(parts)


def verify_served_adapter(base_url: str, served_name: str, adapter_path: str, base_model: str,
                          prompts: list[str], tol: float = 0.05,
                          previous_path: str | None = None) -> dict:
    """Raise `SwapMismatch` if `served_name` does not serve the adapter at `adapter_path`.

    Returns the `measure_served_adapter` dict; `fresh_vs_base` is how far the adapter has moved
    from base (informational: tiny early in training, which is exactly when a text comparison
    against base is meaningless)."""
    out = measure_served_adapter(base_url, served_name, adapter_path, base_model, prompts,
                                 previous_path)
    if out["served_vs_fresh"] > tol:
        raise SwapMismatch(
            f"vLLM's '{served_name}' does not serve the adapter just published "
            f"(first-token distance {out['served_vs_fresh']:.4f} > {tol}; {describe(out)}). "
            f"The swap did not take, so rollouts would come from a model that is not the one "
            f"being trained.", out)
    return out
