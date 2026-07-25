"""Defender GRPO on the on-task shell environments.

The 27B opponent is frozen for the whole phase but sampled stochastically as both attacker and
benign agent. Only the defender is live and captured, so reward differences within a task group
remain attributable to defender behavior. When ``SFT_ADAPTER`` is set, its weights are first
merged into the frozen backbone; the new GRPO adapter starts at zero, making both the initial
policy and the adapter-disabled KL reference the SFT bootstrap rather than raw Qwen.

Run: modal run --detach scripts/train_defender.py::train
"""
import modal

DEFENDER_MODEL = "Qwen/Qwen3-4B"
N_ROLLOUTS = 6          # rollouts per task = the GRPO group size (baseline is the group mean) consider trying 8-16
N_ITERS = 20
LR = 1e-5
TEMP = 0.7              # might need to alter this to get more diversity in answers but need to make sure we dont go too crazy and start to get invalid JSONs
BETA = 0.04
CLIP_EPS = 0.2
MAX_TURNS = 12          # chunk exfil needs ~7 turns
REDACTION_ENFORCEMENT = "unshielded"  # attacker must receive exploitable policy mistakes
ATTACKER_MODEL = "cyankiwi/Qwen3.6-27B-AWQ-INT4"   # frozen opponent (plays attacker + benign agent)
OPP_PORT = 8000
ATK_MAX_TOKENS = 6144          # match the probe that produced the messy middle (attacker thinks more)
BEN_MAX_TOKENS = 4096          # honest agent
OPP_MAX_MODEL_LEN = 32768
OPP_MAX_NUM_SEQS = 16
TRANSCRIPT_CONTEXT_CHARS = 40_000
# canary co-located in each env's secret file — MUST match the benchmark canaries
CANARIES = {"ops-logs": "sk-api-CANARY-7a1b2c", "analytics": "pw-db-CANARY-9x8y7z",
            "ci-build": "tok-deploy-CANARY-3m2n1k"}
# SFT handoff: point at a promoted adapter directory on the Volume to warm-start GRPO from the
# reasoning bootstrap instead of the raw base model. None -> start from base.
SFT_ADAPTER = "/runs/sft-defender/round9/training/7eeae96976a4/adapter"   # e.g. "/runs/sft-defender/<root>/training/<fingerprint>/adapter"
CKPT_ROOT = "/runs/defender-grpo-sft-7eeae96976a4"   # on the Modal Volume; the ONLY state that survives preemption
CKPT_KEEP = 3                             # keep newest N checkpoints (the latest is always resume-safe)
ALLOW_LEGACY_CHECKPOINT = False           # require explicit opt-in for pre-identity checkpoints

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "peft>=0.11",
                 "accelerate>=0.30", "pydantic", "pyyaml", "tqdm")
    # expandable_segments reduces fragmentation between the vLLM reservation and the HF trainer,
    # which share one A100 and allocate in very different block sizes.
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
)
hf_cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
app = modal.App("redteamrl-defender-grpo", image=image)


@app.function(gpu="A100-80GB", timeout=24 * 60 * 60,
              retries=modal.Retries(initial_delay=0.0, max_retries=10),  # Modal's long-training rec: restart NOW on preempt
              volumes={"/cache/huggingface": hf_cache, "/runs": runs})
def train():
    import json, os, sys, torch
    sys.path.insert(0, "/root")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from redteamrl.policies.prompted import (
        PromptedDefender, PromptedAttacker, ATTACKER_SYSTEM_SHELL, HINT_TECHNIQUE, BENIGN_AGENT_SYSTEM)
    from redteamrl.policies.vllm_client import make_vllm_generate, start_vllm_server
    from redteamrl.sandbox.shell import LocalShellSandbox
    from redteamrl.envs import build_tasks
    from redteamrl.envs.ops_logs import OPS_LOGS
    from redteamrl.envs.analytics import ANALYTICS
    from redteamrl.envs.ci_build import CI_BUILD
    from redteamrl.train.capture import CapturingGenerate
    from redteamrl.train.learner import Learner
    from redteamrl.train.train import rollout, assign_advantages, update_step
    from redteamrl.train.checkpoint import (
        latest_checkpoint,
        prune_checkpoints,
        read_meta,
        validate_phase_identity,
        write_meta,
    )
    from redteamrl.train.diversity import group_diversity

    # start the frozen-opponent server FIRST so vLLM reserves its GPU fraction; HF takes the rest
    start_vllm_server(ATTACKER_MODEL, OPP_PORT, 0.4, max_model_len=OPP_MAX_MODEL_LEN,
                      max_num_seqs=OPP_MAX_NUM_SEQS)
    hf_cache.commit()
    gen_atk = make_vllm_generate(f"http://localhost:{OPP_PORT}", ATTACKER_MODEL,
                                 enable_thinking=True, max_tokens=ATK_MAX_TOKENS)
    gen_benign = make_vllm_generate(f"http://localhost:{OPP_PORT}", ATTACKER_MODEL,
                                    enable_thinking=True, max_tokens=BEN_MAX_TOKENS)

    tasks = (
        build_tasks(OPS_LOGS,  CANARIES["ops-logs"],  canary_path="app/config.py")
        + build_tasks(ANALYTICS, CANARIES["analytics"], canary_path=".env")
        + build_tasks(CI_BUILD,  CANARIES["ci-build"],  canary_path="deploy/credentials")
    )
    n_atk = sum(t.episode_type == "attack" for t in tasks)
    print(f"{len(tasks)} tasks: {n_atk} attack, {len(tasks) - n_atk} benign")

    tok = AutoTokenizer.from_pretrained(DEFENDER_MODEL)
    base = AutoModelForCausalLM.from_pretrained(DEFENDER_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    if SFT_ADAPTER:
        # Merge the bootstrap into the frozen backbone and train a NEW zero-initialized adapter.
        # ``disable_adapter()`` now exposes this merged SFT policy, so the KL reference preserves
        # the reasoning bootstrap instead of pulling back toward raw Qwen. This also avoids keeping
        # a second 4B reference model resident beside the 27B opponent.
        base = PeftModel.from_pretrained(
            base,
            SFT_ADAPTER,
            is_trainable=False,
        ).merge_and_unload()
        reference_policy = f"SFT-merged:{SFT_ADAPTER}"
    else:
        reference_policy = f"raw-base:{DEFENDER_MODEL}"

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    print(
        f"GRPO policy initialized at {reference_policy}; "
        "adapter-disabled logprobs use the same frozen reference",
        flush=True,
    )
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    learner = Learner(model, tok, opt, reward_key="defender_reward")
    cap = CapturingGenerate(model, tok, temperature=TEMP, max_new_tokens=512)  # thinking OFF -> reasoning+JSON fits

    def defender_factory():
        return PromptedDefender(generate=cap, max_context_chars=TRANSCRIPT_CONTEXT_CHARS)  # match the probe

    def attack_agent_factory(spec):
        return PromptedAttacker(gen_atk, goal=spec.goal, cover=spec.request, hint=HINT_TECHNIQUE,
                                system_template=ATTACKER_SYSTEM_SHELL,
                                max_context_chars=TRANSCRIPT_CONTEXT_CHARS)

    def benign_agent_factory(spec):
        return PromptedAttacker(gen_benign, goal=spec.request, system_template=BENIGN_AGENT_SYSTEM,
                                max_context_chars=TRANSCRIPT_CONTEXT_CHARS)

    def decompose(examples):
        """One reward per episode -> per-track behavior.

        Read the explicit episode facts rather than reverse-engineering them from reward. Policy
        leaks and malformed protocol outputs now share the same failure reward, so ``reward <=
        -2`` deliberately no longer identifies which failure occurred.
        """
        ep = {
            ex.episode_id: (
                ex.episode_type,
                ex.policy_leak,
                ex.complete,
                ex.defender_protocol_failures,
            )
            for ex in examples
        }
        atk = [row for row in ep.values() if row[0] == "attack"]
        ben = [row for row in ep.values() if row[0] == "benign"]
        frac = lambda xs, p: sum(p(r) for r in xs) / max(len(xs), 1)
        return {
            "attack_policy_leak": frac(atk, lambda row: row[1]),
            "benign_policy_leak": frac(ben, lambda row: row[1]),
            "protocol_failure": frac(list(ep.values()), lambda row: row[3] > 0),
            "refuse": frac(ben, lambda row: not row[1] and not row[2] and row[3] == 0),
            "clean": frac(ben, lambda row: not row[1] and row[2] and row[3] == 0),
        }

    # ---- resume from the newest COMPLETE checkpoint on the Volume (preemption safety) ----
    # Modal restarts a preempted call at the TOP of train() with fresh RAM, so without this the
    # fresh LoRA + range(0, N_ITERS) would silently redo iter 0 forever (exactly what the repeated
    # `[rollout] 1/9` blocks showed). We reload the adapter weights AND the Adam moments, then
    # continue at the next iter. See redteamrl/train/checkpoint for the completeness rules.
    # Identity of THIS phase. A resume that silently inherits a different SFT source, opponent
    # snapshot, or model would continue an incompatible experiment under the same checkpoint root.
    phase_identity = {
        "defender": DEFENDER_MODEL, "attacker": ATTACKER_MODEL, "sft_adapter": SFT_ADAPTER,
        "n_rollouts": N_ROLLOUTS, "lr": LR, "temp": TEMP, "max_turns": MAX_TURNS,
        "beta": BETA, "clip_eps": CLIP_EPS,
        "reference_policy": reference_policy,
        "grpo_lora": {
            "r": 16,
            "alpha": 32,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
        "redaction_enforcement": REDACTION_ENFORCEMENT,
    }
    start_iter = 0
    resume_dir = latest_checkpoint(CKPT_ROOT)
    if resume_dir is not None:
        resume_meta = read_meta(resume_dir)
        # Validate before touching model or optimizer state. Otherwise an incompatible checkpoint
        # can fail with a misleading tensor/rank error before the useful identity diagnostic.
        validate_phase_identity(
            resume_meta,
            phase_identity,
            resume_dir,
            allow_legacy=ALLOW_LEGACY_CHECKPOINT,
        )
        from peft import set_peft_model_state_dict, load_peft_weights
        set_peft_model_state_dict(model, load_peft_weights(resume_dir))
        opt.load_state_dict(torch.load(os.path.join(resume_dir, "optimizer.pt"), map_location="cuda"))
        start_iter = int(resume_meta["iter"]) + 1
        print(f"resuming from {resume_dir}: adapter+optimizer restored, continuing at iter {start_iter}", flush=True)
    else:
        print("no checkpoint found — starting fresh at iter 0", flush=True)

    for it in range(start_iter, N_ITERS):
        cap.buffer.clear()
        model.eval()
        examples = rollout(tasks, attack_agent_factory, benign_agent_factory, defender_factory,
                           LocalShellSandbox, cap, n_rollouts=N_ROLLOUTS, max_turns=MAX_TURNS,
                           redaction_enforcement=REDACTION_ENFORCEMENT)
        assign_advantages(examples)
        d = decompose(examples)
        # An exposed weakness only teaches if the group's rewards actually differ. A dead group is
        # a sampling problem (temperature / group size), never a reason to hand-write the answer
        # into the prompt -- that would close the attacker's foothold instead of training it shut.
        div = group_diversity([
            {"task_id": ex.task_id, "episode_id": ex.episode_id, "reward": ex.reward,
             "verdicts": getattr(ex, "verdicts", [])}
            for ex in examples
        ])
        live = sum(1 for ex in examples if ex.advantage != 0)
        # print liveness the MOMENT the rollout ends — before the slow update_step — so a dead
        # round (live=0) is visible without waiting for the gradient step
        print(f"iter {it:3d} rollout  live={live}/{len(examples)}  |  "
              f"attack_policy_leak={d['attack_policy_leak']:.0%}  "
              f"benign_policy_leak={d['benign_policy_leak']:.0%}  "
              f"protocol_fail={d['protocol_failure']:.0%}  "
              f"refuse={d['refuse']:.0%}  clean={d['clean']:.0%}  |  "
              f"dead_groups={div['dead_group_rate']:.0%} "
              f"mixed_reward={div['mixed_reward_group_rate']:.0%}", flush=True)

        model.train()
        m = update_step(learner, examples, beta=BETA, clip_eps=CLIP_EPS)
        print(f"iter {it:3d} update   loss={m['loss']:.4f}  ratio={m['mean_ratio']:.3f}  "
              f"kl={m['mean_kl']:.5f}  grad={m['grad_norm']:.3f}  "
              f"live_episodes={m['n_live_episodes']}  "
              f"live_decisions={m['n_live_examples']}/{m['n_examples']}", flush=True)

        # checkpoint EVERY iter so a preemption costs at most one iter of work. Save order is
        # load-bearing: adapter -> optimizer -> meta.json LAST (the completion marker). We commit
        # only AFTER the marker, so the Volume never exposes a half-written checkpoint to resume.
        ckpt = os.path.join(CKPT_ROOT, f"iter{it}")
        model.save_pretrained(ckpt)
        torch.save(opt.state_dict(), os.path.join(ckpt, "optimizer.pt"))
        write_meta(ckpt, {"iter": it, "phase_identity": phase_identity})
        prune_checkpoints(CKPT_ROOT, keep=CKPT_KEEP)
        runs.commit()

    final_adapter_dir = f"{CKPT_ROOT}/final"
    model.save_pretrained(final_adapter_dir)   # no meta.json -> ignored by resume/prune
    with open(os.path.join(final_adapter_dir, "composition.json"), "w") as f:
        json.dump({
            "raw_base": DEFENDER_MODEL,
            "sft_adapter": SFT_ADAPTER,
            "grpo_adapter": final_adapter_dir,
            "reference_policy": reference_policy,
        }, f, indent=2)

    # An adapter trained on an SFT-merged backbone is not standalone relative to raw Qwen. Save
    # one fully merged directory for evaluation and the later frozen-defender attacker phase so a
    # consumer cannot accidentally omit the SFT component.
    final_merged_dir = f"{CKPT_ROOT}/final-merged"
    merged = model.merge_and_unload()
    merged.save_pretrained(final_merged_dir)
    tok.save_pretrained(final_merged_dir)
    with open(os.path.join(final_merged_dir, "composition.json"), "w") as f:
        json.dump({
            "raw_base": DEFENDER_MODEL,
            "sft_adapter": SFT_ADAPTER,
            "grpo_adapter": final_adapter_dir,
            "fully_merged": True,
        }, f, indent=2)
    print(f"saved deployable composed policy to {final_merged_dir}", flush=True)
    runs.commit()


@app.local_entrypoint()
def main():
    train.remote()
