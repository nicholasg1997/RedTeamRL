"""~10-minute probe: does a runtime LoRA swap on vLLM actually change the SERVED weights?

Motivation: attacker GRPO crashed at iter 2 because vLLM kept serving the iteration-0 adapter
while the trainer moved on. The original `vllm_lora_probe.py` could not have caught this -- it
swapped in a byte-IDENTICAL copy of the adapter, so it only proved the swap returns 200, never that
the served weights change. This probe swaps in DIFFERENT adapters and checks which one vLLM serves.

Phases (each swap installs a fresh, distinct random adapter):
  0. startup   -- serve adapter A via --lora-modules (sanity: registration works)
  1. OLD swap  -- overwrite the SAME path with B, load_inplace, no unload (the pre-fix code path).
                  Serving A afterwards = bug reproduced (diagnosis confirmed).
  2. NEW swap  -- C at a UNIQUE path via the fixed load_lora_adapter (unload + load). Must serve C.
  3. NEW swap, SAME path -- D overwrites C's path (unload-only; this is train_defender's pattern).
  4. NEW swap again -- E at another unique path. The bug showed on the 2nd+ swap, so repeat it.

vLLM and HF greedy can diverge on near-tied tokens (benign; seen in the GRPO logs), so each vLLM
output is classified by which HF reference (base/A/B/C/D/E) it shares the longest prefix with, over
several prompts, rather than demanding exact equality. Adapters are large random perturbations so
the references are far apart.

Run: modal run scripts/vllm_swap_probe.py
     modal run scripts/vllm_swap_probe.py --model Qwen/Qwen3-8B   # optional: the real model
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

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "peft>=0.11",
                 "accelerate>=0.30", "pydantic", "pyyaml", "tqdm")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
)
hf_cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
app = modal.App("redteamrl-vllm-swap-probe", image=image)


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def closest(output: str, refs: dict[str, list[str]], i: int) -> str:
    """Name of the reference whose i-th output shares the longest prefix with `output`.
    'ambiguous' on a tie for the top score."""
    scores = {name: common_prefix_len(output, outs[i]) for name, outs in refs.items()}
    best = max(scores.values())
    winners = [name for name, s in scores.items() if s == best]
    return winners[0] if len(winners) == 1 else "ambiguous"


@app.function(gpu="A10G", timeout=40 * 60, volumes={"/cache/huggingface": hf_cache})
def probe(model_id: str = DEFAULT_MODEL):
    import shutil, sys, tempfile
    from collections import Counter
    import requests
    import torch
    sys.path.insert(0, "/root")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from redteamrl.policies.vllm_client import (
        load_lora_adapter, start_vllm_server, stop_vllm_server)

    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                device_map="cuda")
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=TARGETS,
                                            task_type="CAUSAL_LM"))
    model.eval()
    rendered = [tok.apply_chat_template([{"role": "user", "content": p}],
                                        add_generation_prompt=True, enable_thinking=False,
                                        tokenize=False) for p in PROMPTS]

    def hf_greedy(adapter_on: bool) -> list[str]:
        outs = []
        for text in rendered:
            ids = tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")
            ctx = model.disable_adapter() if not adapter_on else torch.no_grad()
            with torch.no_grad(), ctx:
                out = model.generate(**ids, do_sample=False, max_new_tokens=MAX_NEW)
            outs.append(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs

    def make_adapter(seed: int, scale: float):
        g = torch.Generator(device="cpu").manual_seed(seed)
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                std = 0.02 if "lora_A" in name else scale
                p.data.copy_(torch.randn(p.shape, generator=g) * std)

    def save_to(path: str):
        model.save_pretrained(path)

    def vllm_greedy() -> list[str]:
        outs = []
        for text in rendered:
            r = requests.post(f"http://localhost:{PORT}/v1/completions",
                              json={"model": LORA_NAME, "prompt": text, "max_tokens": MAX_NEW,
                                    "temperature": 0.0, "add_special_tokens": False},
                              timeout=120)
            r.raise_for_status()
            outs.append(r.json()["choices"][0]["text"])
        return outs

    # ---- build five distinct adapters; raise the perturbation until references separate ----
    refs: dict[str, list[str]] = {"base": hf_greedy(adapter_on=False)}
    scale = 0.05
    used_scale = scale
    for attempt in range(4):
        used_scale = scale          # the scale the references below are computed at
        trial = {}
        for label, seed in zip("ABCDE", (11, 22, 33, 44, 55)):
            make_adapter(seed, used_scale)
            trial[label] = hf_greedy(adapter_on=True)
        names = ["base", *trial]
        all_refs = {"base": refs["base"], **trial}
        distinct = sum(len({all_refs[n][i] for n in names}) == len(names)
                       for i in range(len(PROMPTS)))
        print(f"[setup] scale={used_scale}: {distinct}/{len(PROMPTS)} prompts fully distinct",
              flush=True)
        if distinct >= 3:
            refs.update(trial)
            break
        scale *= 2
    else:
        print("[setup] FAILED to make distinguishable adapters; results below are unreliable.",
              flush=True)
        refs.update(trial)

    def install(label: str, path: str):
        make_adapter({"A": 11, "B": 22, "C": 33, "D": 44, "E": 55}[label], used_scale)
        save_to(path)

    def served_as(phase: str, expect: str) -> bool:
        outs = vllm_greedy()
        labels = [closest(o, refs, i) for i, o in enumerate(outs)]
        verdict, count = Counter(labels).most_common(1)[0]
        ok = verdict == expect
        print(f"[{phase}] vLLM serves '{verdict}' ({count}/{len(labels)} prompts; per-prompt "
              f"{labels}) -- expected '{expect}' -> {'OK' if ok else 'MISMATCH'}", flush=True)
        return ok

    fixed_path = tempfile.mkdtemp(prefix="probe-fixed-")
    install("A", fixed_path)
    proc = None
    results = {}
    try:
        proc = start_vllm_server(model_id, PORT, 0.45, max_model_len=2048, max_num_seqs=8,
                                 lora_name=LORA_NAME, lora_path=fixed_path)
        url = f"http://localhost:{PORT}"
        results["0 startup serves A"] = served_as("0 startup", "A")

        # 1. OLD code path: same name, same (overwritten) path, load_inplace, no unload.
        install("B", fixed_path)
        r = requests.post(f"{url}/v1/load_lora_adapter",
                          json={"lora_name": LORA_NAME, "lora_path": fixed_path,
                                "load_inplace": True}, timeout=300)
        print(f"[1 old swap] load_inplace same path -> {r.status_code} {r.text[:200]!r}",
              flush=True)
        old_fresh = served_as("1 old swap", "B")
        results["1 old swap refreshed (False = bug reproduced)"] = old_fresh

        # 2. NEW code path, unique path.
        unique_c = tempfile.mkdtemp(prefix="probe-c-")
        install("C", unique_c)
        load_lora_adapter(url, LORA_NAME, unique_c)
        results["2 new swap, unique path serves C"] = served_as("2 new unique", "C")

        # 3. NEW code path, SAME path as the previous swap (unload-only; train_defender's pattern).
        install("D", unique_c)
        load_lora_adapter(url, LORA_NAME, unique_c)
        results["3 new swap, same path serves D (unload alone)"] = served_as("3 new same-path", "D")

        # 4. NEW code path again -- the original bug appeared on the 2nd+ swap.
        unique_e = tempfile.mkdtemp(prefix="probe-e-")
        install("E", unique_e)
        load_lora_adapter(url, LORA_NAME, unique_e)
        results["4 repeated new swap serves E"] = served_as("4 new repeat", "E")
    finally:
        stop_vllm_server(proc)
        for d in (fixed_path,):
            shutil.rmtree(d, ignore_errors=True)

    print("\n===== SWAP PROBE SUMMARY =====", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)
    diagnosis = results.get("1 old swap refreshed (False = bug reproduced)") is False
    fix_ok = all(results.get(k) for k in ("2 new swap, unique path serves C",
                                           "4 repeated new swap serves E"))
    same_path_ok = results.get("3 new swap, same path serves D (unload alone)")
    print(f"\n  DIAGNOSIS CONFIRMED (old swap served stale weights): {diagnosis}", flush=True)
    print(f"  FIX VERIFIED for train_attacker (unique path + unload): {fix_ok}", flush=True)
    print(f"  unload ALONE sufficient (train_defender reuses its path): {same_path_ok}", flush=True)
    return results


@app.local_entrypoint()
def main(model: str = DEFAULT_MODEL):
    probe.remote(model)
