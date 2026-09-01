"""Did defender GRPO teach judgment, memorization, or episode-type classification?

The training curve (attack policy leak 78% -> 50% -> 33%, benign flat) is consistent with all
three, and they are not distinguishable from aggregate leak rates. The deterministic probes are:

  matched result probe -- same task, same action, the ONLY difference is whether the credential
    field holds a usable secret or a sentinel. An episode-type classifier scores ~50% on
    `pair_taint_accuracy` however good its type detection is, because it must get BOTH halves
    right. The probe's secrets are also derived independently of the training canaries, so a
    memorizer gains nothing here either.

  call probe -- `od`, `base64` and `cut` each appear on BOTH sides, so "reject the scary verb"
    scores zero on `verb_pair_accuracy`. This is the gate that produced most attack leaks and that
    no per-step label could reach.

Each probe runs over TWO task sets (PLAN.md §7):
  seen      -- ci-build, a TRAINING environment. Held out on secrets and on cases, not on world.
  held_out  -- the reserved environments (PLAN.md §2.9), never trained on by anything.
The GAP between them is the overfitting estimate, a quantity this project has never had. Variants
are tuned so the two sets carry COMPARABLE case counts (held-out has 12 tasks to ci-build's 3),
because resolution is what the exit criterion needs, not variants-per-task.

Served on vLLM, one policy at a time. The previous HF `model.generate` path ran one case at a time
on an idle A100; it also measured a different serving path than train_defender.py uses, so vLLM is
the more faithful measurement as well as the affordable one.

Run: modal run scripts/probe_grpo_adapter.py::evaluate
"""
import modal

DEFENDER_MODEL = "Qwen/Qwen3-4B"
SFT_ADAPTER = "/runs/sft-defender/round9/training/7eeae96976a4/adapter"
# Checkpoint iterN holds the policy AFTER iteration N's update. iter1 (8 gradient steps) is the
# policy that generated the 33% attack-leak rollout in iteration 2.
GRPO_ADAPTER = "/runs/defender-grpo-r9-vllm/iter1"

SEEN_VARIANTS = 30        # 3 ci-build tasks  x 30 x 2 halves = 180 matched cases
HELD_OUT_VARIANTS = 8     # 12 reserved tasks x  8 x 2 halves = 192 matched cases
PORT = 8001
MEM_FRAC = 0.85           # one 4B server, no trainer resident: hand it the card
MAX_MODEL_LEN = 8192      # defender prompts are ~2.5k + 512 generated
WORKERS = 16              # in-flight request count IS vLLM's batch size
MAX_NEW_TOKENS = 512
EVAL_TEMP = 0.0           # deterministic: every difference is signal, not sampling

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "torch>=2.2", "transformers<5", "peft>=0.11", "accelerate>=0.30",
                 "requests", "pydantic", "pyyaml", "tqdm")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
)
hf_cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
app = modal.App("redteamrl-probe-grpo", image=image)

RESULT_KEYS = ("verdict_recall", "taint_cleared_given_attempt", "span_transcription_rate",
               "task_content_preserved", "exact_span_rate", "adversarial_clean_pass_rate",
               "pair_taint_accuracy", "pair_exact_accuracy")
CALL_KEYS = ("reject_recall", "allow_precision", "balanced_accuracy", "verb_pair_accuracy")


def _evaluate_impl(active_servers):
    import json, sys, tempfile
    import torch
    sys.path.insert(0, "/root")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from redteamrl.envs import build_tasks
    from redteamrl.envs.ci_build import CI_BUILD
    from redteamrl.envs.registry import held_out_tasks, HELD_OUT_ENVS
    from redteamrl.policies.vllm_client import (
        make_vllm_generate, start_vllm_server, stop_vllm_server)
    from redteamrl.sft.call_probe import build_call_probe_cases, evaluate_call_probe
    from redteamrl.sft.counterfactuals import build_matched_result_counterfactuals
    from redteamrl.sft.probe import evaluate_matched_probe

    seen = build_tasks(CI_BUILD, "tok-deploy-CANARY-3m2n1k", canary_path="deploy/credentials")
    held = held_out_tasks()
    SETS = {
        "seen": (build_matched_result_counterfactuals(seen, variants_per_task=SEEN_VARIANTS),
                 build_call_probe_cases(seen)),
        "held_out": (build_matched_result_counterfactuals(held, variants_per_task=HELD_OUT_VARIANTS),
                     build_call_probe_cases(held)),
    }
    print(f"reserved envs: {[e.id for e in HELD_OUT_ENVS]}", flush=True)
    for name, (result_cases, call_cases) in SETS.items():
        print(f"{name:9} result={len(result_cases):4}  call={len(call_cases):3}", flush=True)

    def merge_policy(with_grpo):
        """Rebuild the exact stack the trainer serves: base -> merge SFT -> merge GRPO."""
        out = tempfile.mkdtemp(prefix="probe-policy-", dir="/tmp")
        model = AutoModelForCausalLM.from_pretrained(
            DEFENDER_MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
        model = PeftModel.from_pretrained(model, SFT_ADAPTER).merge_and_unload()
        if with_grpo:
            model = PeftModel.from_pretrained(model, GRPO_ADAPTER).merge_and_unload()
        model.save_pretrained(out)
        AutoTokenizer.from_pretrained(DEFENDER_MODEL).save_pretrained(out)
        del model
        return out

    report = {}
    for label, with_grpo in (("sft", False), ("sft+grpo", True)):
        merged = merge_policy(with_grpo)
        # One policy resident at a time: two 4B servers would halve each one's KV cache for no gain,
        # and a stale server surviving a same-container Modal retry is how the last run OOM'd.
        proc = start_vllm_server(merged, PORT, MEM_FRAC, max_model_len=MAX_MODEL_LEN,
                                 max_num_seqs=WORKERS)
        active_servers.append((stop_vllm_server, proc))
        try:
            generate = make_vllm_generate(f"http://localhost:{PORT}", merged,
                                          temperature=EVAL_TEMP, max_tokens=MAX_NEW_TOKENS,
                                          enable_thinking=False)
            print(f"\n===== {label} =====", flush=True)
            report[label] = {}
            for name, (result_cases, call_cases) in SETS.items():
                result = evaluate_matched_probe(result_cases, generate, max_workers=WORKERS)
                call = evaluate_call_probe(call_cases, generate, max_workers=WORKERS)
                report[label][name] = {"result": result, "call": call}
                print(f"  -- {name} --", flush=True)
                for key in RESULT_KEYS:
                    print(f"    result.{key:30} {result[key]:.1%}", flush=True)
                for key in CALL_KEYS:
                    print(f"    call.{key:32} {call[key]:.1%}", flush=True)
        finally:
            stop_vllm_server(proc)
            active_servers.remove((stop_vllm_server, proc))

    print("\n===== GRPO DELTA (sft -> sft+grpo) =====", flush=True)
    for name in SETS:
        print(f"  -- {name} --", flush=True)
        for section, keys in (("result", ("verdict_recall", "taint_cleared_given_attempt",
                                          "task_content_preserved", "adversarial_clean_pass_rate",
                                          "pair_taint_accuracy")),
                              ("call", CALL_KEYS)):
            for key in keys:
                before = report["sft"][name][section][key]
                after = report["sft+grpo"][name][section][key]
                print(f"    {section}.{key:30} {before:6.1%} -> {after:6.1%}  "
                      f"({after - before:+.1%})", flush=True)

    print("\n===== OVERFITTING GAP (seen - held_out, on sft+grpo) =====", flush=True)
    for section, keys in (("result", RESULT_KEYS), ("call", CALL_KEYS)):
        for key in keys:
            gap = (report["sft+grpo"]["seen"][section][key]
                   - report["sft+grpo"]["held_out"][section][key])
            print(f"  {section}.{key:30} {gap:+.1%}", flush=True)

    pair = report["sft+grpo"]["held_out"]["result"]["pair_taint_accuracy"]
    verb = report["sft+grpo"]["held_out"]["call"]["verb_pair_accuracy"]
    print(
        "\nREADING: pair_taint_accuracy is the classifier test -- the two halves share a task and "
        f"an action, so type detection cannot help. On HELD-OUT environments: {pair:.1%}.\n"
        "         The probe's secrets are derived independently of the training canaries, so a "
        "memorizer gains nothing here either.\n"
        f"         verb_pair_accuracy is the 'reject the scary verb' test. Held-out: {verb:.1%}.\n"
        "         A large positive OVERFITTING GAP means the monitor learned the training worlds, "
        "not transferable judgment -- which is the case PLAN.md §2.9 exists to detect.\n"
        "         NOTE: served on vLLM. PLAN.md §4.3's baselines were measured under HF "
        "`model.generate`, so compare within this run, not against those numbers.",
        flush=True,
    )
    with open("/runs/defender-grpo-r9-vllm/probe_iter1.json", "w") as handle:
        json.dump(report, handle)
    runs.commit()


@app.function(gpu="A100-80GB", timeout=90 * 60,
              volumes={"/cache/huggingface": hf_cache, "/runs": runs})
def evaluate():
    """Own the vLLM lifecycle outside the body: a retry in the same container must not inherit
    the previous attempt's resident server."""
    active_servers = []
    try:
        return _evaluate_impl(active_servers)
    finally:
        for stop_server, proc in reversed(active_servers):
            stop_server(proc)


@app.local_entrypoint()
def main():
    evaluate.remote()
