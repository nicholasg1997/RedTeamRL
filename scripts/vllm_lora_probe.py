"""Five-minute probe: does the defender-on-vLLM plan actually work on this stack?

Everything in the vLLM-defender design rests on three claims that cannot be checked offline:
  1. vLLM serves the merged 4B with `--enable-lora` and accepts a runtime adapter swap.
  2. `return_token_ids` returns the EXACT sampled token ids, so GRPO's importance ratio is
     computed on the sequence that was actually generated rather than a retokenization of its text.
  3. A 27B AWQ server and a 4B LoRA server fit on one A100 alongside the HF trainer.

This prints the raw response JSON so the parser can be written against the real schema instead of
a guess. Cheap to run, and a wrong guess here would fail after an hour of rollout.

Run: modal run scripts/vllm_lora_probe.py::probe
"""
import modal

DEFENDER_MODEL = "Qwen/Qwen3-4B"
ATTACKER_MODEL = "cyankiwi/Qwen3.6-27B-AWQ-INT4"
DEF_PORT = 8001
OPP_PORT = 8000
# vLLM sizes its KV cache as (total_gpu_memory * gpu_memory_utilization) MINUS the memory this
# process allocated during profiling. Two servers must therefore start SEQUENTIALLY: if the 27B is
# still allocating while the 4B profiles, the 4B attributes the 27B's growth to itself and computes
# a negative cache budget. The fraction also has to cover the profiling forward pass, which scales
# with max_model_len * max_num_seqs -- that dummy batch, not the weights, is the large term.
DEF_MEM_FRAC = 0.30
OPP_MEM_FRAC = 0.35
DEF_MAX_MODEL_LEN = 8192      # defender prompts are ~2.5k tokens + 512 generated
DEF_MAX_NUM_SEQS = 8

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
app = modal.App("redteamrl-vllm-lora-probe", image=image)


@app.function(gpu="A100-80GB", timeout=45 * 60, volumes={"/cache/huggingface": hf_cache})
def probe():
    import json, os, subprocess, sys, tempfile, time
    import requests
    import torch
    sys.path.insert(0, "/root")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    def gpu_free():
        free, total = torch.cuda.mem_get_info()
        return f"{(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB used"

    # ---- 1. a zero-initialized adapter, exactly as GRPO iteration 0 would hold ----
    tok = AutoTokenizer.from_pretrained(DEFENDER_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        DEFENDER_MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
    lora = LoraConfig(r=16, lora_alpha=32,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    peft_model = get_peft_model(base, lora)
    adapter_v1 = tempfile.mkdtemp(prefix="adapter-v1-")
    peft_model.save_pretrained(adapter_v1)
    del base, peft_model
    print(f"[1] wrote adapter -> {adapter_v1}", flush=True)

    # ---- 2. both servers resident at once: the memory question ----
    print(f"[2] before any server: {gpu_free()}", flush=True)
    def wait(port, name, timeout_s=1800):
        for _ in range(timeout_s // 2):
            try:
                if requests.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                    print(f"[2] {name} healthy on :{port} — {gpu_free()}", flush=True)
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    opp = subprocess.Popen(["vllm", "serve", ATTACKER_MODEL, "--port", str(OPP_PORT),
                            "--gpu-memory-utilization", str(OPP_MEM_FRAC),
                            "--max-model-len", "32768", "--max-num-seqs", "16"])
    dfn = None
    try:
        # SEQUENTIAL: the 4B must not profile while the 27B is still allocating.
        if not wait(OPP_PORT, "27B opponent"):
            print("FAIL: opponent never became healthy", flush=True)
            return
        dfn = subprocess.Popen(["vllm", "serve", DEFENDER_MODEL, "--port", str(DEF_PORT),
                                "--gpu-memory-utilization", str(DEF_MEM_FRAC),
                                "--max-model-len", str(DEF_MAX_MODEL_LEN),
                                "--max-num-seqs", str(DEF_MAX_NUM_SEQS),
                                "--enable-lora", "--max-lora-rank", "16",
                                "--lora-modules", f"defender={adapter_v1}"])
        if not wait(DEF_PORT, "4B defender+LoRA"):
            print("FAIL: defender never became healthy — look for 'Available KV cache memory' "
                  "above. Negative means DEF_MEM_FRAC is too small to cover weights + the "
                  "profiling batch; raise it or lower DEF_MAX_MODEL_LEN/DEF_MAX_NUM_SEQS.",
                  flush=True)
            return
        print(f"[2] both servers up: {gpu_free()}", flush=True)

        # ---- 3. THE critical question: exact token ids back from generation ----
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": "You are a monitor."},
             {"role": "user", "content": "Reply with one short JSON object."}],
            add_generation_prompt=True, enable_thinking=False, tokenize=False,
        )
        local_prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        response = requests.post(
            f"http://localhost:{DEF_PORT}/v1/completions",
            json={"model": "defender", "prompt": prompt, "max_tokens": 64,
                  "temperature": 0.7, "add_special_tokens": False,
                  "return_token_ids": True},
            timeout=120,
        )
        print(f"[3] status={response.status_code}", flush=True)
        payload = response.json()
        print("[3] RAW RESPONSE JSON (write the parser against this):", flush=True)
        print(json.dumps(payload, indent=2)[:4000], flush=True)

        choice = payload.get("choices", [{}])[0]
        text = choice.get("text", "")
        print(f"\n[3] top-level keys: {sorted(payload)}", flush=True)
        print(f"[3] choice keys:    {sorted(choice)}", flush=True)
        print(f"[3] local prompt_ids ({len(local_prompt_ids)}): {local_prompt_ids[:12]}...",
              flush=True)
        # If vLLM returns prompt ids, they MUST match our local tokenization, or the logprobs
        # recomputed under HF would be conditioned on a different prefix than was sampled.
        for key in ("prompt_token_ids", "token_ids"):
            for holder, label in ((payload, "payload"), (choice, "choice")):
                if key in holder:
                    print(f"[3] {label}.{key} = {str(holder[key])[:200]}", flush=True)

        # ---- 4. runtime adapter swap: what happens between GRPO iterations ----
        adapter_v2 = tempfile.mkdtemp(prefix="adapter-v2-")
        os.system(f"cp -r {adapter_v1}/. {adapter_v2}/")
        swap = requests.post(
            f"http://localhost:{DEF_PORT}/v1/load_lora_adapter",
            json={"lora_name": "defender", "lora_path": adapter_v2, "load_inplace": True},
            timeout=120,
        )
        print(f"\n[4] load_lora_adapter (load_inplace) -> {swap.status_code} {swap.text[:300]}",
              flush=True)
        if swap.status_code != 200:
            retry = requests.post(
                f"http://localhost:{DEF_PORT}/v1/load_lora_adapter",
                json={"lora_name": "defender-v2", "lora_path": adapter_v2}, timeout=120)
            print(f"[4] fallback (new name, no load_inplace) -> "
                  f"{retry.status_code} {retry.text[:300]}", flush=True)

        after = requests.post(
            f"http://localhost:{DEF_PORT}/v1/completions",
            json={"model": "defender", "prompt": prompt, "max_tokens": 32, "temperature": 0.0,
                  "add_special_tokens": False, "return_token_ids": True},
            timeout=120)
        print(f"[4] generation after swap -> {after.status_code}", flush=True)

        # ---- 5. headroom for the HF trainer alongside both servers ----
        print(f"\n[5] with both servers: {gpu_free()}", flush=True)
        trainer = AutoModelForCausalLM.from_pretrained(
            DEFENDER_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
        print(f"[5] + HF trainer resident: {gpu_free()}  <- needs headroom for activations",
              flush=True)
        del trainer
        torch.cuda.empty_cache()
    finally:
        for proc in (dfn, opp):
            if proc is None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()


@app.local_entrypoint()
def main():
    probe.remote()
