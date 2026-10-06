"""~15-minute benchmark: why does the attacker engine decode at a fixed ~300ms/step?

Measured in attacker GRPO (2026-10-05): the 8B LoRA attacker engine ran ~3 tok/s PER SEQUENCE
whether 13 or 47 sequences were in flight, on 1 CPU or 8 -- while the co-located 4B defender engine
ran ~19-80 tok/s per sequence. CPU was ruled out (8 cores, 4 used, no change). This isolates the
three remaining differences between the two engines, one variable at a time, on the SAME server
configuration as training (Qwen3-8B, mem 0.42, max_model_len 16384, max_num_seqs 48, LoRA r16):

  A  base model, greedy                  (no LoRA, no sampling)
  B  base model, sampled                 (+ temp 0.7 / top_p 0.8, as the attacker samples)
  C  LoRA adapter, greedy                (+ LoRA kernels only)
  D  LoRA adapter, sampled               (= the attacker's training configuration)
  E  D while a 4B engine decodes on the same GPU   (+ GPU sharing, as in training)

Each runs N_SEQS concurrent requests generating exactly GEN_TOKENS tokens (ignore_eos), and reports
aggregate and per-sequence tok/s. Whichever step collapses per-sequence speed is the bottleneck.

Run: modal run scripts/vllm_decode_bench.py
"""
import modal

ATTACKER = "Qwen/Qwen3-8B"
DEFENDER = "Qwen/Qwen3-4B"
ATK_PORT, DEF_PORT = 8001, 8000
N_SEQS = 48
GEN_TOKENS = 256
PROMPT_WORDS = 400     # ~500-token prompts, like an early attacker turn

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "peft>=0.11",
                 "accelerate>=0.30", "pydantic", "pyyaml", "tqdm")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True", "VLLM_SERVER_DEV_MODE": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
)
hf_cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
app = modal.App("redteamrl-vllm-decode-bench", image=image)


@app.function(gpu="A100-80GB", cpu=8.0, timeout=45 * 60, volumes={"/cache/huggingface": hf_cache})
def bench():
    import sys, tempfile, threading, time
    from concurrent.futures import ThreadPoolExecutor
    import requests
    import torch
    sys.path.insert(0, "/root")
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from redteamrl.policies.vllm_client import start_vllm_server, stop_vllm_server, reset_prefix_cache

    # A real-shaped r16 adapter (B=0 init, so outputs equal base -- kernel cost is what matters).
    # Built on CPU: only its files are needed.
    adapter_dir = tempfile.mkdtemp(prefix="bench-lora-")
    model = AutoModelForCausalLM.from_pretrained(ATTACKER, dtype=torch.bfloat16)
    get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])).save_pretrained(adapter_dir)
    del model

    def run(port, model_name, temperature, label, n=N_SEQS):
        def one(i):
            # Distinct prompts so prefix caching cannot shortcut the prefill.
            prompt = f"[{label} {i}] " + " ".join(f"word{(i * 7 + k) % 997}" for k in range(PROMPT_WORDS))
            payload = {"model": model_name, "prompt": prompt, "max_tokens": GEN_TOKENS,
                       "temperature": temperature, "ignore_eos": True, "return_token_ids": True}
            if temperature > 0:
                payload["top_p"] = 0.8
            r = requests.post(f"http://localhost:{port}/v1/completions", json=payload, timeout=900)
            r.raise_for_status()
            return len(r.json()["choices"][0].get("token_ids") or []) or GEN_TOKENS
        start = time.time()
        with ThreadPoolExecutor(max_workers=n) as pool:
            total = sum(pool.map(one, range(n)))
        dt = time.time() - start
        agg = total / dt
        print(f"[{label}] {n} seqs x {GEN_TOKENS} tok in {dt:6.1f}s -> {agg:7.1f} tok/s aggregate, "
              f"{agg / n:6.2f} tok/s per seq (~{1000 * n / agg:5.0f} ms/step)", flush=True)
        return agg / n

    atk = dfn = None
    results = {}
    try:
        atk = start_vllm_server(ATTACKER, ATK_PORT, 0.42, max_model_len=16384, max_num_seqs=N_SEQS,
                                lora_name="attacker", lora_path=adapter_dir)
        run(ATK_PORT, ATTACKER, 0.0, "warmup", n=4)
        for label, name, temp in (("A base greedy", ATTACKER, 0.0),
                                  ("B base sampled", ATTACKER, 0.7),
                                  ("C lora greedy", "attacker", 0.0),
                                  ("D lora sampled", "attacker", 0.7)):
            reset_prefix_cache(f"http://localhost:{ATK_PORT}")
            results[label] = run(ATK_PORT, name, temp, label)

        # E: the training configuration's co-location -- a 4B engine decoding on the same GPU.
        dfn = start_vllm_server(DEFENDER, DEF_PORT, 0.15, max_model_len=8192, max_num_seqs=N_SEQS)
        run(DEF_PORT, DEFENDER, 0.0, "def warmup", n=4)
        stop_flag = threading.Event()

        def background_load():
            while not stop_flag.is_set():
                run(DEF_PORT, DEFENDER, 0.0, "  (defender load)", n=8)
        loader = threading.Thread(target=background_load, daemon=True)
        loader.start()
        time.sleep(5)
        reset_prefix_cache(f"http://localhost:{ATK_PORT}")
        results["E lora sampled + 4B co-located"] = run(ATK_PORT, "attacker", 0.7, "E co-located")
        stop_flag.set()
        loader.join(timeout=600)
    finally:
        stop_vllm_server(dfn)
        stop_vllm_server(atk)

    print("\n===== DECODE BENCH SUMMARY (tok/s per sequence) =====", flush=True)
    base = results.get("A base greedy")
    for label, v in results.items():
        rel = f"  ({v / base:.0%} of A)" if base else ""
        print(f"  {label:<34} {v:6.2f}{rel}", flush=True)
    print("  training measured: ~3.1 tok/s per seq at ~44 in flight", flush=True)
    return results


@app.local_entrypoint()
def main():
    bench.remote()
