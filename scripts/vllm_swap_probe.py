"""~10-minute probe: does a runtime LoRA swap on vLLM actually change the SERVED weights?

Motivation: attacker GRPO crashed at iter 2 when publish_policy's check found vLLM's greedy output
equal to HF-base and different from HF+adapter. Two explanations fit that evidence:
  (H1) the swap really is stale -- vLLM kept serving an earlier adapter; or
  (H2) the check was a false alarm -- a small adapter flips a near-tied token under HF but not under
       vLLM's kernels, so vLLM-vs-HF disagreement is not evidence of a stale swap.
A third mechanism could produce H1-like symptoms: (H3) the prefix KV cache, computed under the old
adapter, being reused after the swap.

Design: NEVER compare across engines (v1 of this probe did, and random adapters made every
distribution near-flat, so vLLM and HF diverged on token 1 and every verdict came back
'ambiguous'). Instead each adapter X is loaded ONCE under a brand-new name `refX` from its own
directory -- a name vLLM has never seen cannot hit any cache -- and its greedy output is the
reference. The swapped `probe` name must then reproduce that output EXACTLY (same engine, same
kernels, temperature 0, sequential requests). Each phase is queried before and after
/reset_prefix_cache, separating stale weights (H1) from stale prefix KV (H3).

Phases on the `probe` name:
  0. startup  -- A via --lora-modules
  1. OLD swap -- overwrite the SAME path with B, load_inplace, no unload (pre-fix code path)
  2. NEW swap -- C at a UNIQUE path via the fixed load_lora_adapter (unload + load)
  3. NEW swap, SAME path -- D overwrites C's path (unload-only; train_defender's pattern)
  4. NEW swap again -- E at another unique path (the crash appeared on the 2nd+ swap)

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
MAX_NEW = 24
SEEDS = {"A": 11, "B": 22, "C": 33, "D": 44, "E": 55}
LORA_B_STD = 0.05

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "peft>=0.11",
                 "accelerate>=0.30", "pydantic", "pyyaml", "tqdm")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
          "VLLM_SERVER_DEV_MODE": "1",          # exposes /reset_prefix_cache (H3 test)
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
)
hf_cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
app = modal.App("redteamrl-vllm-swap-probe", image=image)


def identify(outs: list[str], refs: dict[str, list[str]]) -> str:
    """Name of the reference whose outputs match `outs` EXACTLY on every prompt, else 'none'.
    Refs are checked to be pairwise distinct before use, so at most one can match."""
    hits = [name for name, ref in refs.items() if ref == outs]
    return hits[0] if len(hits) == 1 else ("none" if not hits else "+".join(hits))


@app.function(gpu="A10G", timeout=40 * 60, volumes={"/cache/huggingface": hf_cache})
def probe(model_id: str = DEFAULT_MODEL):
    import os, shutil, sys, tempfile
    import requests
    import torch
    sys.path.insert(0, "/root")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from redteamrl.policies.vllm_client import (
        load_lora_adapter, start_vllm_server, stop_vllm_server)

    # Adapters are built on CPU: only their saved files matter, HF never generates here.
    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=TARGETS,
                                            task_type="CAUSAL_LM"))
    rendered = [tok.apply_chat_template([{"role": "user", "content": p}],
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

    def greedy(served_name: str) -> list[str]:
        outs = []
        for text in rendered:      # sequential: no batch-composition nondeterminism
            r = requests.post(f"{url}/v1/completions",
                              json={"model": served_name, "prompt": text, "max_tokens": MAX_NEW,
                                    "temperature": 0.0, "add_special_tokens": False},
                              timeout=120)
            r.raise_for_status()
            outs.append(r.json()["choices"][0]["text"])
        return outs

    def reset_prefix_cache() -> int | None:
        try:
            return requests.post(f"{url}/reset_prefix_cache", timeout=60).status_code
        except requests.RequestException as exc:
            print(f"[reset] failed: {exc!r}", flush=True)
            return None

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

        # ---- references: each adapter under a never-before-seen name, from its own dir ----
        refs = {"base": greedy(model_id)}
        for label in SEEDS:
            name = f"ref{label}"
            r = requests.post(f"{url}/v1/load_lora_adapter",
                              json={"lora_name": name, "lora_path": ref_dirs[label]}, timeout=300)
            r.raise_for_status()
            refs[label] = greedy(name)
            again = greedy(name)
            print(f"[ref {label}] deterministic={again == refs[label]}  "
                  f"first prompt -> {refs[label][0][:60]!r}", flush=True)
            requests.post(f"{url}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=60)
        names = list(refs)
        distinct = all(refs[a] != refs[b] for i, a in enumerate(names) for b in names[i + 1:])
        print(f"[setup] references pairwise distinct: {distinct}", flush=True)
        if not distinct:
            print("[setup] UNRELIABLE: two adapters (or an adapter and base) produce identical "
                  "outputs; raise LORA_B_STD.", flush=True)

        def check(phase: str, expect: str) -> dict:
            before = identify(greedy(LORA_NAME), refs)
            status = reset_prefix_cache()
            after = identify(greedy(LORA_NAME), refs)
            print(f"[{phase}] serves '{before}' before prefix-cache reset, '{after}' after "
                  f"(reset -> {status}); expected '{expect}'", flush=True)
            return {"before": before == expect, "after": after == expect,
                    "served_before": before, "served_after": after}

        results["0"] = check("0 startup", "A")

        # 1. OLD code path: same name, same (overwritten) path, load_inplace, no unload.
        install("B", fixed_path)
        r = requests.post(f"{url}/v1/load_lora_adapter",
                          json={"lora_name": LORA_NAME, "lora_path": fixed_path,
                                "load_inplace": True}, timeout=300)
        print(f"[1 old swap] load_inplace same path -> {r.status_code} {r.text[:200]!r}",
              flush=True)
        results["1"] = check("1 old swap", "B")

        # 2. NEW code path, unique path.
        path_c = os.path.join(root, "probe-c")
        install("C", path_c)
        load_lora_adapter(url, LORA_NAME, path_c)
        results["2"] = check("2 new unique", "C")

        # 3. NEW code path, SAME path as the previous swap (train_defender's pattern).
        install("D", path_c)
        load_lora_adapter(url, LORA_NAME, path_c)
        results["3"] = check("3 new same-path", "D")

        # 4. NEW code path again.
        path_e = os.path.join(root, "probe-e")
        install("E", path_e)
        load_lora_adapter(url, LORA_NAME, path_e)
        results["4"] = check("4 new repeat", "E")
    finally:
        stop_vllm_server(proc)
        shutil.rmtree(root, ignore_errors=True)

    def verdict(r: dict) -> str:
        if r["before"]:
            return "fresh"
        if r["after"]:
            return "STALE PREFIX CACHE only (weights fresh)"
        return f"STALE WEIGHTS (served '{r['served_after']}')"

    print("\n===== SWAP PROBE SUMMARY =====", flush=True)
    print(f"  references distinct (probe trustworthy): {distinct}", flush=True)
    for key, label in (("0", "startup A"), ("1", "OLD swap -> B"), ("2", "NEW unique -> C"),
                       ("3", "NEW same-path -> D"), ("4", "NEW repeat -> E")):
        if key in results:
            print(f"  {key} {label:<20} {verdict(results[key])}", flush=True)
    old = results.get("1", {})
    print(f"\n  OLD swap was stale (original diagnosis H1/H3 holds): "
          f"{bool(old) and not old['before']}", flush=True)
    print(f"  NEW swap fresh for train_attacker (2 and 4): "
          f"{all(results.get(k, {}).get('before') for k in ('2', '4'))}", flush=True)
    print(f"  unload alone sufficient (3, train_defender): "
          f"{bool(results.get('3', {}).get('before'))}", flush=True)
    return results


@app.local_entrypoint()
def main(model: str = DEFAULT_MODEL):
    probe.remote(model)
