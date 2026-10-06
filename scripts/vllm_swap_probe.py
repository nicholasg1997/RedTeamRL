"""~10-minute probe: after a runtime LoRA swap, does vLLM serve the NEW adapter on prompts it has
already seen?

Finding (v2 of this probe + vLLM 0.21.0 source): the weights swap fine, but LoRA prefix-cache blocks
are keyed on `lora_name` only (kv_cache_utils._gen_lora_extra_hash_keys). Neither unload nor a new
path evicts them, so a prompt seen before the swap reuses KV computed by the PREVIOUS adapter. The
publish check's fixed prompt and every rollout's shared system prompt are exactly such prompts.
load_lora_adapter now resets the prefix cache after every load; this probe reproduces the bug on
the raw swap and verifies the fixed path.

Measurement: greedy exact-match is useless here -- random adapters make distributions near-flat,
and even a cache hit vs miss flips tokens (v2 saw same-adapter outputs differ). Instead, compare the
first generated token's top-K logprob distribution (a continuous signature that is computed THROUGH
the prefix cache) against references: each adapter loaded once under a never-seen name. Kernel noise
moves the distance a little (the printed noise floor); a different adapter moves it a lot.

Phases on the `probe` name (each preceded by queries that warm the cache under the old adapter):
  0. startup         -- A via --lora-modules
  1. RAW swap to B   -- load_inplace, no reset (the old code path)      -> expect STALE (not B)
     then reset_prefix_cache alone                                     -> expect B
  2. load_lora_adapter, unique path   -> C
  3. load_lora_adapter, SAME path     -> D (train_defender's pattern)
  4. load_lora_adapter, unique path   -> E (the crash appeared on the 2nd+ swap)

Run: modal run scripts/vllm_swap_probe.py
"""
import modal

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"   # LoRA-management mechanism is model-independent; keep it cheap
PORT = 8011
LORA_NAME = "probe"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PROMPTS = [
    "Reply with one short JSON object.",
    "Name three colors.",
    "Write one sentence about the ocean.",
    "What is 17 times 3? Answer briefly.",
    "Give a one-word greeting.",
]
# Like training's shared system prompt: long enough to fill several 16-token KV blocks, so most of
# each prompt is a prefix-cache hit after the first query -- which is where staleness lives.
SYSTEM = ("You are a careful assistant working inside a sandboxed research environment. Read each "
          "request fully, think about what is being asked, and answer concisely. Prefer short, "
          "well-formed outputs. If a request asks for structured data, emit valid JSON and nothing "
          "else. Do not add commentary before or after the answer unless explicitly asked to.")
TOP_K = 20
SEEDS = {"A": 11, "B": 22, "C": 33, "D": 44, "E": 55}
LORA_B_STD = 0.05

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "peft>=0.11",
                 "accelerate>=0.30", "pydantic", "pyyaml", "tqdm")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
          "VLLM_SERVER_DEV_MODE": "1",          # exposes /reset_prefix_cache
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
)
hf_cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
app = modal.App("redteamrl-vllm-swap-probe", image=image)


def dist_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Mean |logprob difference| over the union of two top-K maps. A token missing from one map is
    floored just below that map's K-th logprob (it was not in the top K, so it is at most that)."""
    floor_p, floor_q = min(p.values()) - 1.0, min(q.values()) - 1.0
    keys = set(p) | set(q)
    return sum(abs(p.get(k, floor_p) - q.get(k, floor_q)) for k in keys) / len(keys)


def signature_distance(a: list[dict], b: list[dict]) -> float:
    return sum(dist_distance(x, y) for x, y in zip(a, b)) / len(a)


def nearest(sig: list[dict], refs: dict[str, list[dict]]) -> tuple[str, float, float]:
    """(closest ref name, its distance, distance to the runner-up)."""
    ranked = sorted((signature_distance(sig, r), n) for n, r in refs.items())
    return ranked[0][1], ranked[0][0], ranked[1][0]


@app.function(gpu="A10G", timeout=40 * 60, volumes={"/cache/huggingface": hf_cache})
def probe(model_id: str = DEFAULT_MODEL):
    import os, shutil, sys, tempfile
    import requests
    import torch
    sys.path.insert(0, "/root")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from redteamrl.policies.vllm_client import (
        load_lora_adapter, reset_prefix_cache, start_vllm_server, stop_vllm_server)

    # Adapters are built on CPU: only their saved files matter, HF never generates here.
    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=TARGETS,
                                            task_type="CAUSAL_LM"))
    rendered = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": p}],
                                        add_generation_prompt=True, enable_thinking=False,
                                        tokenize=False) for p in PROMPTS]

    def install(label: str, path: str):
        g = torch.Generator(device="cpu").manual_seed(SEEDS[label])
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                std = 0.02 if "lora_A" in name else LORA_B_STD
                p.data.copy_((torch.randn(p.shape, generator=g) * std).to(p.dtype))
        model.save_pretrained(path)

    url = f"http://localhost:{PORT}"

    def signature(served_name: str) -> list[dict]:
        sig = []
        for text in rendered:
            r = requests.post(f"{url}/v1/completions",
                              json={"model": served_name, "prompt": text, "max_tokens": 1,
                                    "temperature": 0.0, "logprobs": TOP_K,
                                    "add_special_tokens": False}, timeout=120)
            r.raise_for_status()
            sig.append(r.json()["choices"][0]["logprobs"]["top_logprobs"][0])
        return sig

    root = tempfile.mkdtemp(prefix="swap-probe-")
    ref_dirs = {}
    for label in SEEDS:
        ref_dirs[label] = os.path.join(root, f"ref-{label}")
        install(label, ref_dirs[label])
    fixed_path = os.path.join(root, "probe-fixed")
    install("A", fixed_path)

    proc = None
    results = {}
    try:
        proc = start_vllm_server(model_id, PORT, 0.45, max_model_len=2048, max_num_seqs=8,
                                 lora_name=LORA_NAME, lora_path=fixed_path)

        # ---- references: each adapter under a never-seen name; 2nd query (a cache HIT) gives
        # the noise floor between the cached and uncached compute paths ----
        refs = {"base": signature(model_id)}
        noise = []
        for label in SEEDS:
            name = f"ref{label}"
            r = requests.post(f"{url}/v1/load_lora_adapter",
                              json={"lora_name": name, "lora_path": ref_dirs[label]}, timeout=300)
            r.raise_for_status()
            refs[label] = signature(name)
            noise.append(signature_distance(refs[label], signature(name)))
            requests.post(f"{url}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=60)
        names = list(refs)
        separation = min(signature_distance(refs[a], refs[b])
                         for i, a in enumerate(names) for b in names[i + 1:])
        noise_floor = max(noise)
        trustworthy = separation > 5 * noise_floor
        print(f"[setup] noise floor (cache hit vs miss, same adapter) = {noise_floor:.4f}; "
              f"closest pair of references = {separation:.4f}; trustworthy={trustworthy}",
              flush=True)

        def check(phase: str, expect: str) -> bool:
            served, d, runner_up = nearest(signature(LORA_NAME), refs)
            ok = served == expect
            print(f"[{phase}] serves '{served}' (dist {d:.4f}, runner-up {runner_up:.4f}); "
                  f"expected '{expect}' -> {'OK' if ok else 'STALE/WRONG'}", flush=True)
            return ok

        results["0 startup serves A"] = check("0 startup", "A")

        # 1. RAW swap (old code path): same name+path, load_inplace, NO cache reset.
        install("B", fixed_path)
        r = requests.post(f"{url}/v1/load_lora_adapter",
                          json={"lora_name": LORA_NAME, "lora_path": fixed_path,
                                "load_inplace": True}, timeout=300)
        print(f"[1 raw swap] load_inplace -> {r.status_code}", flush=True)
        results["1a raw swap serves B (False = bug reproduced)"] = check("1a raw swap", "B")
        reset_prefix_cache(url)
        results["1b after cache reset serves B (True = cache was the cause)"] = check(
            "1b raw swap + reset", "B")

        # 2-4. The fixed path (load_lora_adapter resets the cache itself).
        path_c = os.path.join(root, "probe-c")
        install("C", path_c)
        load_lora_adapter(url, LORA_NAME, path_c)
        results["2 fixed swap, unique path serves C"] = check("2 fixed unique", "C")

        install("D", path_c)
        load_lora_adapter(url, LORA_NAME, path_c)
        results["3 fixed swap, same path serves D"] = check("3 fixed same-path", "D")

        path_e = os.path.join(root, "probe-e")
        install("E", path_e)
        load_lora_adapter(url, LORA_NAME, path_e)
        results["4 fixed swap again serves E"] = check("4 fixed repeat", "E")
    finally:
        stop_vllm_server(proc)
        shutil.rmtree(root, ignore_errors=True)

    print("\n===== SWAP PROBE SUMMARY =====", flush=True)
    print(f"  probe trustworthy (refs separated >5x noise floor): {trustworthy}", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)
    diagnosis = (results.get("1a raw swap serves B (False = bug reproduced)") is False and
                 results.get("1b after cache reset serves B (True = cache was the cause)") is True)
    fixed = all(results.get(k) for k in ("2 fixed swap, unique path serves C",
                                          "3 fixed swap, same path serves D",
                                          "4 fixed swap again serves E"))
    print(f"\n  DIAGNOSIS (stale prefix cache, weights fine): {diagnosis}", flush=True)
    print(f"  FIX VERIFIED (attacker + defender swap patterns): {fixed}", flush=True)
    return results


@app.local_entrypoint()
def main(model: str = DEFAULT_MODEL):
    probe.remote(model)
