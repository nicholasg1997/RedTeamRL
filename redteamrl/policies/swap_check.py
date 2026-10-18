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


def verify_served_adapter(base_url: str, served_name: str, adapter_path: str, base_model: str,
                          prompts: list[str], tol: float = 0.05) -> dict:
    """Raise if `served_name` does not serve the adapter at `adapter_path`.

    Returns {"served_vs_fresh", "fresh_vs_base"} -- the second is how far the adapter has moved
    from base (informational: tiny early in training, which is exactly when a text comparison
    against base is meaningless)."""
    fresh = f"swapcheck-{next(_counter)}"
    r = requests.post(f"{base_url.rstrip('/')}/v1/load_lora_adapter",
                      json={"lora_name": fresh, "lora_path": adapter_path}, timeout=300)
    r.raise_for_status()
    try:
        served_sig = first_token_signature(base_url, served_name, prompts)
        fresh_sig = first_token_signature(base_url, fresh, prompts)
        base_sig = first_token_signature(base_url, base_model, prompts)
    finally:
        requests.post(f"{base_url.rstrip('/')}/v1/unload_lora_adapter",
                      json={"lora_name": fresh}, timeout=60)
    out = {"served_vs_fresh": signature_distance(served_sig, fresh_sig),
           "fresh_vs_base": signature_distance(fresh_sig, base_sig)}
    if out["served_vs_fresh"] > tol:
        raise RuntimeError(
            f"vLLM's '{served_name}' does not serve the adapter just published "
            f"(first-token distance {out['served_vs_fresh']:.4f} > {tol} vs a fresh load of the same "
            f"file; adapter-vs-base distance {out['fresh_vs_base']:.4f}). The swap did not take, so "
            f"rollouts would come from a model that is not the one being trained.")
    return out
