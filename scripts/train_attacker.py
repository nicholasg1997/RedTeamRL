"""Attacker GRPO — phase 3 of the loop (SFT -> defender GRPO -> ATTACKER GRPO -> repeat).

*** RUN THE VIABILITY GATE FIRST: modal run scripts/attacker_viability.py::check ***
A win rate outside 0.15-0.85 means dead groups and no gradient -- 20 iterations of nothing. If the
attacker is TOO WEAK, fall back to the SFT-only defender as the opponent (drop DEFENDER_GRPO_ADAPTER
to None) before reaching for a bigger attacker.

The opponent below is the FULL measured stack (base + SFT + GRPO). Sampling config is thinking-OFF
at 2048: the viability check measured both arms and found identical win rates, with thinking-6144
truncating 29.5% of its actions. Matching the 27B's sampling config was never going to make this an
equivalent opponent anyway -- the model differs, which is the variable that dominates.

Mirror image of train_defender.py: the DEFENDER is now the frozen opponent on vLLM and the
attacker is the live, captured, trainable model. `rollout(..., trained_side="attacker")` tags
captured examples with `attacker_reward`, so exactly one side is learning and every bit of
within-group reward variance is attributable to it.

Two things to watch, both printed every iter:
  * dead_group_rate — GRPO's baseline is the group mean, so identical rewards across a group give
    zero advantage. An attacker that never succeeds (or always succeeds) against the current
    defender learns nothing. That is a sampling/opponent-strength problem, not a prompt problem.
  * The defender snapshot is FROZEN for the whole run. Promote a new one between phases, never
    mid-run, or the attacker is chasing a moving target and the credit assignment is lost.

Run: modal run --detach scripts/train_attacker.py::train
"""
import modal

ATTACKER_MODEL = "Qwen/Qwen3-8B"     # trainable attacker; must be an unquantized base for LoRA
# The attacker SFT adapter to START GRPO FROM (the PLAN §6 kl=0 ratchet). When set, it is MERGED
# into the base before the fresh training LoRA, so `disable_adapter()` returns the SFT policy as
# GRPO's KL reference -- GRPO improves on SFT instead of discarding it. None = start from raw base
# (B2 was: GRPO ignored SFT entirely). Set to the promoted v4 adapter AFTER re-running the gate.
ATTACKER_SFT_ADAPTER = None          # e.g. "/runs/attacker-sft-v4/adapter"
DEFENDER_MODEL = "Qwen/Qwen3-4B"     # frozen opponent this phase
# The frozen opponent is base -> SFT -> GRPO, merged in that order (same stack as
# probe_grpo_adapter.py and attacker_viability.py). Set GRPO to None for an SFT-only opponent.
DEFENDER_SFT_ADAPTER = "/runs/sft-defender/round9/training/7eeae96976a4/adapter"
DEFENDER_GRPO_ADAPTER = "/runs/defender-grpo-r9-vllm/iter1"
# GRPO group size; larger groups buy within-group variance. 12 (v7): at a 15% win rate a group
# of 8 is all-loss 27% of the time, a group of 12 only 14%.
N_ROLLOUTS = 12
N_ITERS = 20
# 2e-5 (was 1e-5) to go with mean-centred advantages: dropping the std division shrinks a
# 1-win-in-8 group's winner advantage 2.65 -> 1.18 (~2.2x), which at 1e-5 would quietly halve
# the step on the win signal. Doubling restores it there, while repeat-only groups (std ~0.02-0.1)
# now step ~20-50x smaller -- the intended effect. The ratio sat at ~0.998 after 4 epochs, far
# from the 0.2 clip, so there is headroom. 3e-5 (2026-10-18) with INNER_EPOCHS 4 -> 2.
# 5e-5 (v7.2): 3e-5 moved the policy too little (v7: kl_iter 0.0009-0.0065, KL to the reference flat);
# 1e-4 too much and noisily (v7.1: 1 then 4 steps before the brake, kl_iter 0.054 then 0.37). Most of
# v7.1's overshoot was Adam's warm-up, which WARMUP_STEPS now absorbs; the KL rule adapts from here.
LR = 5e-5
# Linear LR ramp over the run's first optimizer steps (one full iteration at MINIBATCHES x INNER_EPOCHS).
WARMUP_STEPS = 16
# 0.7 is what attacker_viability measured 16.7% at (make_vllm_generate's default). 1.0 was
# chosen to buy within-group variance, but it bought the opposite: iteration 0 came back
# 100% dead groups at a 0% win rate. Exploration you cannot convert into a win is not
# exploration. Match the measured config first, then reintroduce deltas one at a time.
TEMP = 0.7
BETA = 0.04             # KL weight against the frozen reference
CLIP_EPS = 0.2          # PPO trust region; only bites once inner epochs move the ratio
# Gradient steps per rollout batch; see train_defender.py. Watch `epoch_ratios` for clip saturation.
# 2, not 4 (2026-10-18): the update is compute-bound (~21 min, ~80% of an iteration) and at 4 epochs
# the ratio reached only 1.0003 -- nowhere near the 0.2 clip -- so epochs 3-4 were small repeat
# steps. Two epochs at a higher LR cover similar ground for ~half the update compute.
INNER_EPOCHS = 2
# v7: one optimizer step per MINIBATCH, not per epoch. Pre-v7 the update took ~2 Adam steps per
# iteration (one per epoch over the whole batch) and the attacker barely moved: KL to the
# reference 0.003 after five v6 iterations, held-out at the base rate (11.7%). 8 x 2 = 16 steps.
MINIBATCHES = 8
# Per-iteration behaviour->new KL the LR rule steers toward (redteamrl/train/lr_rule.py); a
# minibatch past 2x this is not stepped and ends the update (the brake). LR above is the INITIAL
# learning rate; the live one is checkpoint state, moved by the rule after every update.
TARGET_KL = 0.01
# --overfit-task: GRPO groups per iteration on the single task (no restarts, no replacements).
OVERFIT_GROUPS = 4
# Micro-batches up to this many padded tokens skip activation checkpointing (store activations
# instead of recomputing them in backward, ~25% of an epoch). 0 = always checkpoint.
# 4096 OOM'd (v4 arm 2, iter 0, 2026-10-18): the trainer reached 33.8 GiB -- ~17 GiB past the 16.4
# GiB of weights -- before a 4096-token batch's stored activations fit. The ~2.7 MB/token estimate
# that chose 4096 left out LoRA's per-module dropout copies and fp32 norm upcasts. Re-enable only
# from the measured `peak=` on the update line, never from an estimate. Not in phase_identity:
# checkpointing changes memory, not math (tests/test_batched_update.py), so resume is unaffected.
NO_CHECKPOINT_MAX_TOKENS = 0
MAX_TURNS = 12
REDACTION_ENFORCEMENT = "unshielded"  # the attacker must receive exploitable policy mistakes
DEF_PORT = 8000
ATK_PORT = 8001
# Frozen 4B: ~8GB of weights, prompts ~2.5k + 512 generated, so its KV need is small.
DEF_MEM_FRAC = 0.15
# Live 8B: ~16.4GB of weights + KV. gpu_memory_utilization is a HARD UP-FRONT RESERVATION --
# vLLM takes it and never returns it -- so every point here is a point the HF trainer cannot have.
# 0.45 left the trainer only ~31.6GB and it OOM'd. Measured KV cache usage on this engine peaked at
# 3.6% of a 19GB pool (~0.7GB), so 0.45 was ~25x over-provisioned: 0.30 still leaves ~7.4GB of KV,
# about 10x the observed peak, and hands the trainer ~12GB back.
# 0.30 was set to survive the OOM BEFORE gradient checkpointing existed. Checkpointing cut the
# trainer's need to ~22.7GB, leaving ~21GB idle. A 15k-token late-episode request holds 2.06GB
# of KV, so 0.30's 7.4GB pool capped LATE-episode concurrency at ~3.6 requests -- utilisation
# collapsed exactly as transcripts grew. 0.42 gives ~8.2 and still leaves the trainer +11GB.
ATK_MEM_FRAC = 0.42
ATK_MAX_MODEL_LEN = 16384
ATK_MAX_NUM_SEQS = 48      # >= ROLLOUT_WORKERS so concurrent episodes batch
# MEASURED 2026-08-31 (attacker_viability): thinking-6144 and no-thinking-2048 gave
# statistically identical win rates (18.8% vs 16.7%, n=48, one episode apart) -- but
# thinking-6144 truncated 29.5% of its actions mid-<think> vs 0.4% here. A truncated
# generation is a MAX-LENGTH sequence, and GRPO weights all completion tokens equally, so
# ~60% of the gradient would come from generations that emitted no action at all.
ATK_MAX_TOKENS = 2048
BEN_MAX_TOKENS = 1024
DEF_MAX_NEW_TOKENS = 512
DEF_MAX_MODEL_LEN = 8192   # defender prompts are ~2.5k + 512 generated; frees room
                           # for the attacker's larger context (was 32768)
DEF_MAX_NUM_SEQS = 48      # matches ROLLOUT_WORKERS; its 24.7k-token KV pool queues the excess
TRANSCRIPT_CONTEXT_CHARS = 40_000
N_GEN_ENVS = 12         # procedurally generated, name-varied training envs (mirrors sft_attacker)
# 102, not 101: the SFT phase collected at rev 101. Sharing it makes SFT targets and GRPO rollouts
# draw BYTE-IDENTICAL canaries, so a policy that memorized the SFT secrets would score a false
# "ratchet is moving" (audit weakness). A distinct revision keeps the two phases' secrets disjoint.
CANARY_REVISION = 102
CANARY_SEED = 0
# The checkpoint root is derived per-run inside _train_impl (it encodes arm + coef so the two arms
# never share a checkpoint lineage). Fresh base: generated envs + SFT adapter + arm/coef are all in
# phase_identity.
# Keys where this run DELIBERATELY diverges from the 2026-08-31 gate. That gate measured raw base
# 8B vs a temp-0.7 defender on the THIN envs at rev 101 -- this run changes all of that (starts from
# the SFT adapter, generated envs, greedy defender, rev 102). The 16.7% no longer predicts it, so
# attacker_viability MUST be re-run on THIS config (set ATTACKER_ADAPTER + generated envs there) and
# MEASURED_GATE_CONFIG refreshed. Until then these are acknowledged so the script can start; each is
# retired by that re-gate, not left here permanently.
# "path_normalised" post-dates the 2026-08-31 gate (Task 1 landed the sandbox-root normaliser
# after it was measured), so it is acknowledged rather than matched against the gate.
ACKNOWLEDGED_CONFIG_DELTAS = frozenset(
    {"attacker_sft_adapter", "env_set", "defender_temperature", "canary_revision",
     "path_normalised"})
# Episodes run concurrently ACROSS ALL TASKS. vLLM batches by in-flight request count, so the old
# task-at-a-time loop pinned both engines at a queue of ~1 (measured: 27B at 61 tok/s, the 4B
# starved to 1.7). Keep this <= the engines' max_num_seqs so requests batch rather than queue.
# 48, not 16 (2026-10-05): at 16 the attacker engine decoded ~13 seqs at ~49 tok/s TOTAL (~4/seq)
# with its KV pool peaking at 7.6% -- decode-bound small batches (230W vs 330W in the update).
# A decode step costs ~the same at batch 13 or 40 (dominated by reading the weights), so 3x the
# in-flight episodes ~triples rollout throughput. Wall-clock only: not in phase_identity, so a
# run resumes across this change. If late-episode KV ever saturates, vLLM preempts (slower, safe).
ROLLOUT_WORKERS = 48
# Benign episodes complete in ~2 turns and then idled to MAX_TURNS under the default fixed
# horizon. A bounded post-completion window gives equal safety exposure at a fraction of the cost.
POST_COMPLETION_TURNS = 4
# Vary the FROZEN defender's phrasing per task so the attacker cannot overfit one
# wording of its opponent. Attacker-side styles are NOT varied: the attacker is the
# trainee, and its own prompt is what the gradient is conditioned on.
# OFF for the first run: viability measured 16.7% against the FIXED DEFENDER_SYSTEM, and a
# rotating opponent prompt is an unvalidated delta from that. The defender is frozen this
# phase, so the pool is opponent diversity (nice to have), not the anti-fingerprint
# property it provides when the DEFENDER is the trainee. Turn back on once ~17% reproduces.
USE_PROMPT_POOL = False
CKPT_KEEP = 3
ALLOW_LEGACY_CHECKPOINT = False

# --- Arm 3 repeat-penalty experiment (spec 2026-10-04 tree-mc-grpo s13) ---
# ARM 2 = plain episode-GRPO (REPEAT_PENALTY_COEF = 0.0); ARM 3 = + exact-repeat penalty.
# Run arms together at EQUAL generated tokens. Sweep the coef over the set below across runs.
# SCALE INTERACTION (why these values): a losing episode scores -0.1 (attacker_reward) and the
# per-episode step penalty totals 0.1; a win scores ~+1.0. At coef=0.1 one repeat (-0.1) already
# equals the loss magnitude, so among LOSING episodes the penalty becomes the dominant gradient --
# which is the upper edge of "testing whether discouraging repeats helps winning" before it tips
# into "training directly on don't-repeat". Hence the sweep is centred lower than the win/loss
# scale, not above it.
# Arm (2 or 3) and the repeat-penalty coef are now the `train()` entrypoint flags
# (defaults arm=3, coef=0.05) so the arm 2/3 sweep runs from one unedited file:
#   modal run --detach scripts/train_attacker.py::train --arm 2 --repeat-penalty-coef 0.0
#   modal run --detach scripts/train_attacker.py::train --arm 3 --repeat-penalty-coef 0.05
# CKPT_ROOT is derived per-run inside _train_impl from the chosen arm/coef.
REPEAT_PENALTY_SWEEP = (0.02, 0.05, 0.1)  # the coef must be one of these; ignored when arm == 2
# Held-out loop metric uses stalled_turn_rate (observation-based), NOT the penalty's action_key.
# n >= 128 so a 3pp win-rate difference is resolvable (at n=32 the per-eval SE ~6pp hides it).
# Eval every k iters (not every iter): single-iter win rates at n=128 have SE ~3pp and are read
# only via the trailing-window mean, so a full eval every iteration burns temp-0 rollouts for
# points we have pre-committed never to read individually (spec s13 review point 3).
# 256 (2026-10-20): evals are now read PAIRED against the base policy's per-seed record
# (eval/heldout_record.py); seeds 1000-1127 are unchanged, so a 128-seed record still pairs.
HELDOUT_EVAL_EPISODES = 256
HELDOUT_EVAL_EVERY = 3             # iterations between held-out evals (plus the final iteration)
PATH_NORMALISED = True             # Task 1 landed; both arms see normalised observations.
# Mean-centred advantages (r - mean), NOT std-normalised (2026-10-06). Arm 3 iter 0 showed why:
# all-loss groups differing by ONE repeat (-0.30 vs -0.35, std 0.02) got +-1..2 advantages -- as
# strong as a win -- and the push grows as a learning policy's groups tighten. In reward units a
# one-repeat difference stays a 0.05 signal and the penalty coefficient means what it says.
ADVANTAGE_STD_NORMALIZE = False
# Bumped whenever the algorithm or the attacker's inputs change, so a run can never resume onto a
# checkpoint chain trained under different rules (phase_identity would refuse it anyway; a fresh
# root makes it a clean start instead of an error). v1 = std-normalised advantages (arm 2 iters
# 0-1, arm 3 iter-0 rollout). v2 = mean-centred advantages + batched update. v3 = the attacker
# sees its own last ATTACKER_HISTORY_TURNS raw outputs in its prompt. v4 = 2 inner epochs, LR 3e-5,
# old_lp taken from epoch 0, checkpoint recompute skipped for small micro-batches, and the update
# hyperparameters (inner_epochs, beta, clip_eps) are now in phase_identity. v5 = adaptive task
# allocation + restarts from earlier win / near-miss states (spec 2026-10-18): who gets a fresh
# group vs a restart group changes every iteration's episode distribution. v6 = dynamic sampling
# (dead base groups get bounded replacement groups), revisit backoff, adaptive restart depth.
# v7 = 16 minibatch steps/iteration with a target-KL brake and a KL-adaptive LR, G=12, restarts
# from the decisive turn (depth 1.0), graduation, same-iteration near-miss banking (spec
# 2026-10-20).
LINEAGE = "v7.2"
# The attacker's last K raw outputs (prose + action), kept in full in its OWN prompt so it stops
# re-deriving the same conclusions each turn; older ones are dropped, a runaway output keeps its
# tail. Never shown to the defender. Changes the attacker's inputs, so it is in phase_identity.
ATTACKER_HISTORY_TURNS = 3

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "peft>=0.11",
                 "accelerate>=0.30", "pydantic", "pyyaml", "tqdm")
    # expandable_segments reduces fragmentation between the vLLM reservations and the HF trainer,
    # which share one A100 and allocate in very different block sizes.
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          # Required for /v1/load_lora_adapter: without it vLLM does not register the route at all
          # and the swap 404s. Startup --lora-modules works regardless, so the server looks healthy.
          "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
          # Exposes /reset_prefix_cache. vLLM 0.21 keys LoRA prefix-cache blocks on the adapter
          # NAME, so every swap must evict them or old-adapter KV is reused (load_lora_adapter).
          "VLLM_SERVER_DEV_MODE": "1"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
)
hf_cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
app = modal.App("redteamrl-attacker-grpo", image=image)

# --split-defender: memory fraction for the defender when it has an L4 (24GB) to itself. 4B bf16 is
# ~7.5GB of weights, so 0.85 leaves ~12GB of KV -- ~3.5x the 3.4GB it got co-located on the A100.
REMOTE_DEF_GPU = "L4"
REMOTE_DEF_MEM_FRAC = 0.85


def _merge_frozen_defender() -> str:
    """Merge the frozen defender's adapters into its backbone; return a servable dir (or the base id).

    The adapters must be MERGED before serving -- declaring them and serving the raw base would
    silently train the attacker against the wrong (unpromoted) defender. Shared by the co-located
    path and RemoteDefender so both serve byte-identical weights."""
    import tempfile
    import torch
    adapters = [a for a in (DEFENDER_SFT_ADAPTER, DEFENDER_GRPO_ADAPTER) if a]
    if not adapters:
        return DEFENDER_MODEL
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    merged_dir = tempfile.mkdtemp(prefix="frozen-defender-", dir="/tmp")
    model = AutoModelForCausalLM.from_pretrained(DEFENDER_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cpu")
    for adapter in adapters:                        # order matters: SFT then GRPO
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(DEFENDER_MODEL).save_pretrained(merged_dir)
    print(f"serving frozen defender = base + {' + '.join(adapters)}", flush=True)
    return merged_dir


# The frozen defender on its own GPU, reached by Modal RPC (not a public URL: the server runs with
# dev endpoints enabled). Stateless per call -- transcript in, verdict out -- so it serves plain
# rollouts and forked tree branches alike, and scales separately from the attacker (max_containers).
# max_containers=1 so autoscaling can never multiply the bill. scaledown_window outlasts an update
# phase (~22 min) so it does not cold-start every iteration. A future DEFENDER-training variant would
# add a method that calls load_lora_adapter (which resets the prefix cache) on this server.
@app.cls(gpu=REMOTE_DEF_GPU, cpu=4.0, timeout=24 * 60 * 60, max_containers=1,
         scaledown_window=30 * 60, volumes={"/cache/huggingface": hf_cache, "/runs": runs})
@modal.concurrent(max_inputs=64)
class RemoteDefender:
    @modal.enter()
    def start(self):
        import sys
        sys.path.insert(0, "/root")
        from redteamrl.policies.vllm_client import make_vllm_generate, start_vllm_server
        served = _merge_frozen_defender()
        self.proc = start_vllm_server(served, DEF_PORT, REMOTE_DEF_MEM_FRAC,
                                      max_model_len=DEF_MAX_MODEL_LEN,
                                      max_num_seqs=DEF_MAX_NUM_SEQS)
        # Same sampling as the co-located path: greedy, thinking off (see the comment there).
        self.gen = make_vllm_generate(f"http://localhost:{DEF_PORT}", served,
                                      enable_thinking=False, temperature=0.0,
                                      max_tokens=DEF_MAX_NEW_TOKENS)
        print(f"RemoteDefender ready on {REMOTE_DEF_GPU}", flush=True)

    @modal.method()
    def generate(self, system: str, messages: list[dict]) -> str:
        return self.gen(system, messages)

    @modal.exit()
    def stop(self):
        from redteamrl.policies.vllm_client import stop_vllm_server
        stop_vllm_server(getattr(self, "proc", None))


def _group_size_suffix(n_rollouts: int) -> str:
    """Checkpoint-root suffix for a non-default group size, so a G=12 run never resumes a G=8 one."""
    return "" if n_rollouts == N_ROLLOUTS else f"-g{n_rollouts}"


def _train_impl(active_servers, arm, repeat_penalty_coef, until_iter=N_ITERS,
                skip_final_update=False, split_defender=False, eval_at_end=False,
                n_rollouts=N_ROLLOUTS, baseline_eval=False, max_turns=MAX_TURNS,
                overfit_task=""):
    import json, os, random, sys, tempfile, time, torch
    sys.path.insert(0, "/root")

    # v7.2 turns the repeat penalty ON: training losses repeated observations ~3-5x more than wins
    # (v7.1 iters 0-1: stalled 40% vs 8-12%), and the 20-turn baseline's extra turns were 52% stalled.
    assert arm == 3, "v7.2 pins arm 3 (repeat penalty); see REPEAT_PENALTY_SWEEP"
    # --max-turns only re-sizes the BASELINE eval (measuring what a longer budget buys the untrained
    # policy). For training it would change the episode distribution, the reward's turn penalty and
    # the viability-gate config, i.e. a new lineage -- a deliberate decision, never a flag typo.
    if max_turns != MAX_TURNS and not baseline_eval:
        raise ValueError(f"--max-turns {max_turns} is baseline-only; training runs MAX_TURNS="
                         f"{MAX_TURNS} (change the constant for a new lineage)")
    coef = repeat_penalty_coef if arm == 3 else 0.0   # arm 2 ignores the coef
    if coef not in REPEAT_PENALTY_SWEEP:
        raise ValueError(f"repeat_penalty_coef {coef} is not one of {REPEAT_PENALTY_SWEEP}")
    ckpt_root = f"/runs/attacker-grpo-{LINEAGE}-arm{arm}-coef{coef}"
    ckpt_root += _group_size_suffix(n_rollouts)       # group size is a lineage flag
    if split_defender:
        # Separate lineage: the defender on L4 kernels can flip near-tied greedy tokens vs the
        # A100, i.e. a (slightly) different opponent -- never mix the two in one checkpoint chain.
        ckpt_root += "-splitdef"
    if overfit_task:
        # The overfit test: can the pipeline master ONE training task it trains on directly? If not,
        # the failure is in the pipeline, not research difficulty. Its own lineage, never resumed
        # into (or from) the full-task run.
        ckpt_root += f"-overfit-{overfit_task}"

    # Preflight, before ~10 minutes of model loading and two server startups. vLLM only registers
    # /v1/load_lora_adapter when this is set, and publish_policy's swap 404s without it -- correct
    # (a stale adapter must never be served) but discovered far too late to be cheap.
    if os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "").lower() not in ("1", "true"):
        raise RuntimeError(
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING is not set on the image. vLLM will not expose "
            "/v1/load_lora_adapter, so the per-iteration adapter swap cannot work. Add it to the "
            "image .env({...}) block.")
    if os.environ.get("VLLM_SERVER_DEV_MODE", "") != "1":
        raise RuntimeError(
            "VLLM_SERVER_DEV_MODE=1 is not set on the image. vLLM will not expose "
            "/reset_prefix_cache, and without it every adapter swap reuses prefix KV computed by "
            "the previous adapter. Add it to the image .env({...}) block.")

    # The gate's number only transfers if training runs what the gate measured. Three knobs once
    # diverged silently and cost an iteration that produced zero gradient.
    from redteamrl.train.config_guard import MEASURED_GATE_CONFIG, assert_matches_gate
    assert_matches_gate(MEASURED_GATE_CONFIG, {
        "attacker_model": ATTACKER_MODEL,
        "defender_model": DEFENDER_MODEL,
        "defender_sft_adapter": DEFENDER_SFT_ADAPTER,
        "defender_grpo_adapter": DEFENDER_GRPO_ADAPTER,
        "atk_temperature": TEMP,
        "atk_max_tokens": ATK_MAX_TOKENS,
        "atk_enable_thinking": False,
        "max_turns": MAX_TURNS,
        "redaction_enforcement": REDACTION_ENFORCEMENT,
        "canary_revision": CANARY_REVISION,
        "defender_prompt": "pool" if USE_PROMPT_POOL else "fixed",
        "attacker_sft_adapter": ATTACKER_SFT_ADAPTER,
        "env_set": "generated",
        "defender_temperature": 0.0,
        "path_normalised": PATH_NORMALISED,
    }, acknowledged=ACKNOWLEDGED_CONFIG_DELTAS)
    print("config matches the viability gate "
          f"({MEASURED_GATE_CONFIG['arm']}, measured {MEASURED_GATE_CONFIG['measured_win_rate']:.1%})",
          flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from redteamrl.policies.prompted import (
        PromptedDefender, PromptedAttacker, ATTACKER_SYSTEM_SHELL, HINT_TECHNIQUE,
        HINT_FOUNDATIONAL_V2,
        BENIGN_AGENT_SYSTEM)
    from redteamrl.policies.vllm_client import (
        load_lora_adapter, make_vllm_generate, served_lora_path, start_vllm_server,
        stop_vllm_server)
    from redteamrl.policies.swap_check import beats_noise, describe, measure_served_adapter
    from redteamrl.sandbox.shell import LocalShellSandbox
    from redteamrl.policies.prompt_pool import (
        DEFENDER_STYLES, PROMPT_POOL_VERSION, select_style)
    from redteamrl.envs.generate import build_generated_tasks
    from redteamrl.envs.registry import assert_training_split
    from redteamrl.train.capture import VLLMCapturingGenerate
    from redteamrl.train.learner import Learner, prepare_for_long_context_training
    from redteamrl.train.train import rollout, assign_advantages, tactic_summary, update_step
    from redteamrl.train.diversity import group_diversity, reward_summary
    from redteamrl.train.allocation import AllocParams, GroupPlan, TaskStats, allocate
    from redteamrl.train.banks import StartStateBank
    from redteamrl.train.episode_store import read_episodes
    from redteamrl.train.v5_finalize import DEFAULT_MAX_NEAR_MISSES, bank_episodes
    from redteamrl.train.v7_finalize import finalize_v7, v7_identity
    from redteamrl.train.lr_rule import LrRule, next_lr
    from redteamrl.train.replacement import load_or_plan, plan_replacements
    from redteamrl.train.metrics_log import MetricsLog
    from redteamrl.train.iter_metrics import attacker_metrics, training_stall
    from redteamrl.train.train import tactic_counts
    from redteamrl.sft.canary import CANARY_SCHEME_VERSION, randomize_task_canary
    from redteamrl.train.checkpoint import (
        latest_checkpoint,
        prune_checkpoints,
        read_meta,
        validate_phase_identity,
        write_meta,
    )

    import shutil
    if split_defender:
        # The frozen defender runs on its OWN L4 (RemoteDefender, below). Co-located, its
        # compute-heavy prefills time-sliced the attacker's short decode steps: ~22 tok/s/seq alone
        # vs ~3 in training. Kick off its cold start NOW so it overlaps this container's ~10 min of
        # model loading; the first rollout call blocks on it only if it is still starting.
        remote_defender = RemoteDefender()
        defender_warmup = remote_defender.generate.spawn(
            "You are a monitor.", [{"role": "user", "content": "Reply with one short JSON object."}])
        served_defender = "remote-L4"

        def gen_def(system, messages):
            return remote_defender.generate.remote(system, messages)
        print("frozen defender: REMOTE (L4, RemoteDefender) -- warming up in parallel", flush=True)
    else:
        # Frozen DEFENDER on vLLM first so it reserves its GPU fraction; HF attacker takes the rest.
        served_defender = _merge_frozen_defender()
        active_servers.append((stop_vllm_server, start_vllm_server(
            served_defender, DEF_PORT, DEF_MEM_FRAC, max_model_len=DEF_MAX_MODEL_LEN,
            max_num_seqs=DEF_MAX_NUM_SEQS)))
        hf_cache.commit()
        # temperature=0.0: the defender is the frozen OPPONENT, not the trainee. At 0.7 sibling
        # rollouts with identical attacker actions can draw different verdicts, so part of the
        # within-group reward variance -- which IS the GRPO advantage signal -- would be opponent
        # coin-flips mis-attributed to the attacker. Greedy makes it a deterministic function of
        # the transcript.
        gen_def = make_vllm_generate(f"http://localhost:{DEF_PORT}", served_defender,
                                     enable_thinking=False, temperature=0.0,
                                     max_tokens=DEF_MAX_NEW_TOKENS)

    # Train on the SAME procedurally generated, name-varied envs as the SFT phase so names/formats
    # cannot predict attack-vs-benign (the generator is the name-invariance substrate). Per-episode
    # canaries are randomized below; the per-env placeholder only needs to be unique.
    tasks = assert_training_split([
        task
        for seed in range(N_GEN_ENVS)
        for task in build_generated_tasks(seed, f"tok-gen-CANARY-{seed:03d}")
    ])
    # Only attack tasks carry an attacker reward; benign episodes have no attacker to train.
    tasks = [task for task in tasks if task.episode_type == "attack"]
    if overfit_task:
        tasks = [task for task in tasks if task.id == overfit_task]
        if not tasks:
            raise ValueError(f"--overfit-task {overfit_task!r} is not a training attack task")
    print(f"{len(tasks)} attack tasks (generated envs x {N_GEN_ENVS})", flush=True)
    task_by_id = {task.id: task for task in tasks}
    # v7: restarts begin AT the decisive turn (k = key_turn) and widen earlier only once a group
    # can finish from there (spec s4.2).
    alloc = (AllocParams(depth_start=1.0, restart_fraction_max=0.0, max_replacements=0,
                         max_fresh_per_task=OVERFIT_GROUPS) if overfit_task
             else AllocParams(depth_start=1.0, near_miss_offset=1, near_miss_min_turns_left=3))
    # GRPO group slots per iteration: one per task normally; OVERFIT_GROUPS on the single task.
    n_slots = OVERFIT_GROUPS if overfit_task else len(tasks)
    lr_rule = LrRule(initial=LR, target_kl=TARGET_KL)

    def episode_task_for(iteration):
        """A fresh high-entropy secret per (iteration, slot, rollout) -- never repeated.

        Keying on rollout_index alone replants the SAME six secrets every iteration, so a policy
        could earn the whole reward by memorizing six strings and generalize to nothing. Including
        the iteration makes every episode a secret the model has not seen, so improvement has to
        transfer. `rollout_index` is slot * n_rollouts + r, and v6 replacement slots run past the
        base slots, so the stride reserves room for them; a task that holds two groups in
        one iteration still draws distinct secrets. Still a pure function of its arguments, so a
        resumed run reproduces exactly the canaries its banked episodes were generated with.
        """
        def episode_task(spec, rollout_index):
            return randomize_task_canary(
                spec, iteration * (n_slots + alloc.max_replacements) * n_rollouts
                + rollout_index, CANARY_SEED,
                CANARY_REVISION)
        return episode_task

    tok = AutoTokenizer.from_pretrained(ATTACKER_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        ATTACKER_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    # kl=0 ratchet (PLAN §6): merge the SFT adapter INTO the base, then stack a fresh training LoRA.
    # disable_adapter() then returns the SFT policy, so GRPO's KL reference is the SFT model, not raw
    # base -- GRPO starts where SFT left off instead of throwing it away (fixes B2).
    if ATTACKER_SFT_ADAPTER:
        from peft import PeftModel as _PeftModel
        base = _PeftModel.from_pretrained(base, ATTACKER_SFT_ADAPTER).merge_and_unload()
        print(f"attacker GRPO starts from SFT adapter {ATTACKER_SFT_ADAPTER} (merged into base)",
              flush=True)
    else:
        print("attacker GRPO starts from RAW BASE (ATTACKER_SFT_ADAPTER=None) -- SFT not wired in",
              flush=True)
    lora = LoraConfig(r=16, lora_alpha=32,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    # 15k-token sequences (40k-char transcript + 2048 completion) save ~41 GiB of activations
    # across 36 layers without this -- more than the whole trainer budget. See the helper for
    # why enable_input_require_grads is not optional under PEFT.
    model = prepare_for_long_context_training(get_peft_model(base, lora))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    learner = Learner(model, tok, opt, reward_key="attacker_reward")
    # ---- serve the SAME policy on vLLM: base backbone + the live GRPO adapter ----
    # An HF `model.generate` trainee is why the first attempt ran at ~7% GPU utilisation: every
    # ROLLOUT_WORKER thread calls generate() on one model, the GIL and CUDA streams serialise them
    # into batch-1 generations, and the opponent's engine idles at `Running: 0 reqs` waiting for
    # work. vLLM batches by in-flight request count, which is what makes concurrency pay.
    adapter_serve_dir = tempfile.mkdtemp(prefix="grpo-live-adapter-", dir="/tmp")
    model.save_pretrained(adapter_serve_dir)
    # Started LAST: the defender engine and the HF trainer are both resident and static by now, so
    # this engine's memory profiling measures only its own allocation.
    active_servers.append((stop_vllm_server, start_vllm_server(
        ATTACKER_MODEL, ATK_PORT, ATK_MEM_FRAC, max_model_len=ATK_MAX_MODEL_LEN,
        max_num_seqs=ATK_MAX_NUM_SEQS, lora_name="attacker", lora_path=adapter_serve_dir,
    )))

    _last_serve_dir = {"path": adapter_serve_dir}   # the startup adapter is the first 'previous'

    def publish_policy(tag):
        """Push the current adapter to vLLM and PROVE the served policy matches the trained one.

        A swap that silently no-ops would keep generating from a stale policy while HF reports a
        healthy KL -- rollouts and gradients would quietly describe different models. That is
        what happened once, and the cause was the PREFIX CACHE, not the weights: vLLM 0.21 keys
        LoRA KV blocks on the adapter name, so this check's fixed prompt (and every rollout's
        shared system prompt) reused KV computed before the swap. load_lora_adapter now resets
        the prefix cache after every load (see its docstring, and scripts/vllm_swap_probe.py for
        the reproduction). The unique dir per publish is kept as defense in depth.
        """
        serve_dir = tempfile.mkdtemp(prefix="grpo-live-adapter-", dir="/tmp")
        model.save_pretrained(serve_dir)
        load_lora_adapter(f"http://localhost:{ATK_PORT}", "attacker", serve_dir)
        # Verify vLLM against ITSELF (policies/swap_check.py): the served name must match a fresh,
        # never-cached load of the same file. The old vLLM-vs-HF greedy comparison raised "serving
        # the BASE policy" on a working swap (v3 arm 2 iter 0: "I'm" vs "I am", ratio=1.000) --
        # after a small update the adapter is ~base and two stacks break near-ties differently.
        check_prompts = [
            tok.apply_chat_template([{"role": "system", "content": "You are an agent."},
                                     {"role": "user", "content": u}],
                                    add_generation_prompt=True, enable_thinking=False,
                                    tokenize=False)
            for u in ("Reply with one short JSON object.", "List the files you would inspect first.",
                      "State your next action in one sentence.")]
        # 1. DETERMINISTIC check: vLLM must have the file just written registered under
        #    "attacker". With the prefix-cache reset (raises on failure, inside load_lora_adapter)
        #    and no load_inplace (pinned by tests), this covers every stale-swap cause we have seen.
        registered = served_lora_path(f"http://localhost:{ATK_PORT}", "attacker")
        if registered is None:
            print(f"[{tag}] WARNING: vLLM reports no path for 'attacker'; registration "
                  f"could not be verified", flush=True)
        elif os.path.realpath(registered) != os.path.realpath(serve_dir):
            raise RuntimeError(f"vLLM serves 'attacker' from {registered}, not the adapter just "
                               f"published ({serve_dir}); rollouts would use the wrong policy")
        # 2. DIAGNOSTIC only: the first-token signature. After a small update consecutive adapters
        #    differ by less than this measurement's own noise (v5 iter 1 raised twice on a working
        #    swap, served-vs-itself repeat > served-vs-fresh), so it warns, never stops the run.
        previous = _last_serve_dir["path"]
        d = measure_served_adapter(f"http://localhost:{ATK_PORT}", "attacker", serve_dir,
                                   ATTACKER_MODEL, check_prompts, previous_path=previous)
        flag = ("  WARNING: differs from a fresh load beyond the check's own noise"
                if beats_noise(d) else "")
        print(f"[{tag}] served policy registered at {serve_dir} ({describe(d)}){flag}",
              flush=True)
        if previous is not None:
            shutil.rmtree(previous, ignore_errors=True)
        _last_serve_dir["path"] = serve_dir

    # Thinking OFF. The 27B that produced the defender's benchmark numbers ran thinking-ON at
    # 6144, but this is an 8B: it does not close its think block inside that budget, and the
    # viability check measured no win-rate benefit for a 29.5% invalid-action rate.
    cap = VLLMCapturingGenerate(
        base_url=f"http://localhost:{ATK_PORT}", model="attacker", tokenizer=tok,
        temperature=TEMP, max_new_tokens=ATK_MAX_TOKENS,
    )

    defender_styles = sorted(DEFENDER_STYLES)

    def defender_factory_for(iteration):
        def defender_factory(spec):
            """Style is fixed within a (task, iteration) -- i.e. within a GRPO GROUP.

            Varying it per rollout would put opponent phrasing into the within-group reward
            variance, and that variance is the advantage signal: it must stay attributable to the
            attacker's sampled actions, not to which defender wording it happened to draw. Across
            tasks and iterations the pool is still sampled ~uniformly.
            """
            template = (DEFENDER_STYLES[
                            defender_styles[select_style(spec.id, 0, iteration, len(defender_styles))]]
                        if USE_PROMPT_POOL else None)
            return PromptedDefender(generate=gen_def, system_template=template,
                                    max_context_chars=TRANSCRIPT_CONTEXT_CHARS)
        return defender_factory

    # v7.2+: foundational hint (principles, no named tactic). Part of phase_identity so a hint
    # change forks the lineage and a resume never mixes two prompt versions.
    ATTACKER_HINT = HINT_FOUNDATIONAL_V2
    ATTACKER_HINT_VERSION = "foundational-v2"
    def attack_agent_factory(spec):
        # The LIVE model. HINT_TECHNIQUE stays as-is: freezing the hint keeps improvements
        # attributable to learning rather than to us writing better attacks by hand.
        return PromptedAttacker(cap, goal=spec.goal, cover=spec.request, hint=ATTACKER_HINT,
                                system_template=ATTACKER_SYSTEM_SHELL,
                                max_context_chars=TRANSCRIPT_CONTEXT_CHARS,
                                history_turns=ATTACKER_HISTORY_TURNS)

    def benign_agent_factory(spec):
        raise AssertionError("attacker phase runs attack tasks only")

    phase_identity = {
        "attacker": ATTACKER_MODEL,
        "attacker_sft_adapter": ATTACKER_SFT_ADAPTER,
        "defender": DEFENDER_MODEL,
        "defender_sft_adapter": DEFENDER_SFT_ADAPTER,
        "defender_grpo_adapter": DEFENDER_GRPO_ADAPTER,
        "atk_max_tokens": ATK_MAX_TOKENS,
        "atk_enable_thinking": False,
        "use_prompt_pool": USE_PROMPT_POOL,
        "prompt_pool_version": PROMPT_POOL_VERSION,
        "n_rollouts": n_rollouts,
        "lr": LR,
        "temp": TEMP,
        "max_turns": MAX_TURNS,
        "redaction_enforcement": REDACTION_ENFORCEMENT,
        "arm": arm,
        "repeat_penalty_coef": coef,
        "path_normalised": PATH_NORMALISED,
        "advantage_std_normalized": ADVANTAGE_STD_NORMALIZE,
        "attacker_history_turns": ATTACKER_HISTORY_TURNS,
        # Update hyperparameters: changing any of them changes the algorithm, so a resume across
        # one would silently continue a chain trained under different rules.
        "inner_epochs": INNER_EPOCHS, "beta": BETA, "clip_eps": CLIP_EPS,
        "lineage": LINEAGE,
    }
    phase_identity.update(v7_identity(alloc, DEFAULT_MAX_NEAR_MISSES))
    phase_identity.update(lr_rule.as_identity())
    phase_identity.update({"minibatches": MINIBATCHES, "heldout_eval_every": HELDOUT_EVAL_EVERY,
                           "brake_rule": "mean2x-or-single10x-v1", "lr_warmup_steps": WARMUP_STEPS,
                           "attacker_hint_version": ATTACKER_HINT_VERSION})
    if overfit_task:
        phase_identity.update({"overfit_task": overfit_task, "overfit_groups": OVERFIT_GROUPS})
    if split_defender:
        # Only added when split, so the co-located path's identity (and its existing checkpoints)
        # is unchanged. The opponent's hardware is part of the episode distribution.
        phase_identity["defender_placement"] = f"remote-{REMOTE_DEF_GPU}"
    start_iter = 0
    # Baseline mode measures the UNTRAINED policy, so it must never load a lineage's weights.
    resume_dir = None if baseline_eval else latest_checkpoint(ckpt_root)
    if resume_dir is not None:
        resume_meta = read_meta(resume_dir)
        validate_phase_identity(
            resume_meta,
            phase_identity,
            resume_dir,
            allow_legacy=ALLOW_LEGACY_CHECKPOINT,
        )
        from peft import set_peft_model_state_dict, load_peft_weights
        set_peft_model_state_dict(model, load_peft_weights(resume_dir))
        opt.load_state_dict(torch.load(os.path.join(resume_dir, "optimizer.pt"),
                                       map_location="cuda"))
        start_iter = int(resume_meta["iter"]) + 1
        # .get so checkpoints written before this key existed still resume (count restarts from 0).
        cumulative_gen_tokens = int(resume_meta.get("cumulative_gen_tokens", 0))
        print(f"resuming from {resume_dir} at iter {start_iter}", flush=True)
    else:
        cumulative_gen_tokens = 0
        print("baseline eval: checkpoints ignored" if baseline_eval
              else "no checkpoint found — starting fresh at iter 0", flush=True)

    # --- v5 allocation state ---------------------------------------------------------------
    # Banks live under the checkpoint root (one directory per task) with the source records they
    # point at. Per-task fresh-start stats ride in each checkpoint's meta, so a resume rebuilds the
    # allocation inputs from the last COMPLETED iteration.
    banks = {task.id: StartStateBank(os.path.join(ckpt_root, "banks", task.id)) for task in tasks}
    bank_record_dir = os.path.join(ckpt_root, "bank-records")
    saved_stats = {d["task_id"]: TaskStats.from_dict(d) for d in
                   (read_meta(resume_dir).get("task_stats", []) if resume_dir is not None else [])}
    task_stats = {task.id: saved_stats.get(task.id, TaskStats(task.id)) for task in tasks}
    # Per-entry restart depth (v6): moves only at finalize, persisted in each checkpoint's meta.
    restart_depths: dict[str, float] = (
        dict(read_meta(resume_dir).get("restart_depths", {})) if resume_dir is not None else {})
    # v7 checkpoint state: graduated bank entries (never drawn again) and the live learning rate
    # the KL-adaptive rule chose after the last completed update.
    resume_meta_v7 = read_meta(resume_dir) if resume_dir is not None else {}
    graduated: set[str] = set(resume_meta_v7.get("graduated", []))
    current_lr: float = float(resume_meta_v7.get("lr", lr_rule.initial))
    optimizer_steps: int = int(resume_meta_v7.get("optimizer_steps", 0))   # drives the LR warmup

    def plans_for(iteration, rollout_dir):
        """This iteration's group plan, persisted beside its rollout.

        A resume mid-iteration replays the banked episodes, which only line up with the SAME
        groups. Recomputing could differ (banks and stats move as iterations finalize), so the
        first computation is saved and every later one loads it."""
        path = os.path.join(rollout_dir, "plans.json")
        if os.path.exists(path):
            with open(path) as handle:
                return [GroupPlan.from_dict(d) for d in json.load(handle)]
        plans = allocate(list(task_stats.values()), banks, n_slots, iteration,
                         random.Random(f"alloc-{LINEAGE}-{iteration}"), alloc,
                         depths=restart_depths, graduated=graduated)
        os.makedirs(rollout_dir, exist_ok=True)
        with open(path + ".tmp", "w") as handle:
            json.dump([plan.to_dict() for plan in plans], handle)
        os.replace(path + ".tmp", path)
        return plans

    # Arm 3 subtracts coef * exact-repeat-count from each episode's reward (coef 0.0 = arm 2,
    # the identity). Built once; the secret it keys on comes from each episode's spec.forbidden[0],
    # which _run_one_episode passes to the transform -- no canary recovery here.
    from redteamrl.train.repeat_penalty import make_repeat_penalty
    reward_transform = make_repeat_penalty(coef)

    # --- held-out eval (greedy, temp-0, n disjoint seeds) ---
    from concurrent.futures import ThreadPoolExecutor
    from redteamrl.harness.episode import run_episode
    from redteamrl.attack_sft.eval_metrics import mean_stalled_turn_rate, dead_observation_rate
    from redteamrl.eval.heldout_record import make_record, save_record
    # build_generated_tasks / randomize_task_canary / CANARY_SEED / CANARY_REVISION are imported
    # and defined above.
    HELDOUT_SEED_START = 1000

    # The held-out defender below is FIXED (system_template=None). That is only faithful to the
    # training defender while the prompt pool is off; if the pool were enabled, training would
    # rotate the defender prompt while this eval stayed fixed -- a silent A/B bias. Fail loudly.
    assert not USE_PROMPT_POOL, (
        "held-out eval uses a FIXED defender prompt; if USE_PROMPT_POOL is enabled, mirror "
        "defender_factory_for here or the arm 2/3 comparison is biased")

    # Mirrors `cap` exactly (same base_url/model/tokenizer/max_new_tokens) but greedy, so the
    # held-out proportion is a clean function of the policy, not a sample of a temperature.
    eval_gen = VLLMCapturingGenerate(
        base_url=f"http://localhost:{ATK_PORT}", model="attacker", tokenizer=tok,
        temperature=0.0, max_new_tokens=ATK_MAX_TOKENS,
    )

    def _heldout_attack_task(seed):
        # Same build-label scheme AND the same randomize_task_canary path as training, so the
        # planted secret is drawn from the identical distribution -- only the (disjoint) seed/id
        # differs. Otherwise a different-shaped target would contaminate the A/B win-rate gap
        # (spec s13 review point 1). gen-{seed} ids with seed>=1000 are disjoint from training
        # (range(N_GEN_ENVS)), so these canaries never collide with a training canary.
        tasks = build_generated_tasks(seed, f"tok-gen-CANARY-{seed:04d}")
        attack = next(t for t in tasks if t.episode_type == "attack")
        return randomize_task_canary(attack, 0, CANARY_SEED, CANARY_REVISION)

    def _heldout_seeds(n):
        # One definition, used by the eval AND by the per-seed records: a record's seed list must be
        # exactly the seeds its outcomes came from (pool.map keeps order), or pairing is garbage.
        return list(range(HELDOUT_SEED_START, HELDOUT_SEED_START + n))

    def _run_heldout_eval(n):
        seeds = _heldout_seeds(n)

        def one(seed):
            # Mirror attack_agent_factory / defender_factory_for exactly (same system prompts, same
            # served model names), differing ONLY in the greedy attacker generate; the defender is
            # the frozen temp-0 opponent gen_def already serving on DEF_PORT.
            spec = _heldout_attack_task(seed)
            attacker = PromptedAttacker(eval_gen, goal=spec.goal, cover=spec.request,
                                        hint=ATTACKER_HINT, system_template=ATTACKER_SYSTEM_SHELL,
                                        max_context_chars=TRANSCRIPT_CONTEXT_CHARS,
                                        history_turns=ATTACKER_HISTORY_TURNS)
            defender = PromptedDefender(generate=gen_def, system_template=None,
                                        max_context_chars=TRANSCRIPT_CONTEXT_CHARS)
            sandbox = LocalShellSandbox()
            try:
                return run_episode(spec, attacker, defender, sandbox, max_turns=max_turns,
                                   redaction_enforcement=REDACTION_ENFORCEMENT)
            finally:
                sandbox.close()

        with ThreadPoolExecutor(max_workers=ROLLOUT_WORKERS) as pool:
            return list(pool.map(one, seeds))

    if baseline_eval:
        # The UNTRAINED policy's held-out rate: the anchor every lineage's held-out number is read
        # against (none was ever measured -- v4 skipped its eval). Nothing has been published yet,
        # so vLLM still serves the startup adapter: a freshly initialised LoRA (B = 0), i.e.
        # exactly the base model, through the same served name, prompts, greedy decode and frozen
        # defender a training run's held-out eval uses. Prove that before measuring.
        registered = served_lora_path(f"http://localhost:{ATK_PORT}", "attacker")
        if (registered is None
                or os.path.realpath(registered) != os.path.realpath(adapter_serve_dir)):
            raise RuntimeError(f"baseline eval must measure the fresh startup adapter "
                               f"({adapter_serve_dir}), but vLLM serves {registered}")
        heldout = _run_heldout_eval(HELDOUT_EVAL_EPISODES)
        wins = sum(r.outcome == "leak" for r in heldout)
        win_rate = wins / len(heldout)
        stalled = mean_stalled_turn_rate([r.steps for r in heldout])
        dead = sum(dead_observation_rate(r.steps) for r in heldout) / len(heldout)
        print(f"baseline heldout  n={len(heldout)}  win_rate={win_rate:.1%}  "
              f"stalled_rate={stalled:.1%}  dead_rate={dead:.1%}", flush=True)
        # Per-seed, so any later checkpoint can be compared to it PAIRED (eval/heldout_record.py).
        turns_tag = "" if max_turns == MAX_TURNS else f"-t{max_turns}"   # 12-turn name unchanged
        save_record(os.path.join("/runs/attacker-heldout-baseline",
                                 f"base-n{len(heldout)}{turns_tag}.json"),
                    make_record(_heldout_seeds(len(heldout)), [r.outcome for r in heldout],
                                turns=[r.n_turns for r in heldout], policy="base",
                                stalled_rate=stalled, dead_rate=dead,
                                attacker=ATTACKER_MODEL,
                                attacker_sft_adapter=ATTACKER_SFT_ADAPTER,
                                defender_sft_adapter=DEFENDER_SFT_ADAPTER,
                                defender_grpo_adapter=DEFENDER_GRPO_ADAPTER,
                                max_turns=max_turns, atk_max_tokens=ATK_MAX_TOKENS))
        runs.commit()
        return

    # Per-process CPU every 60s: a single-threaded process pinned at ~100% (a vLLM scheduler loop,
    # an API server, or the GIL-bound rollout harness) is the leading suspect for the attacker
    # engine's ~3 tok/s/seq (vs ~10 co-located in isolation). See redteamrl/train/cpu_monitor.py.
    from redteamrl.train.cpu_monitor import start_cpu_monitor
    start_cpu_monitor(interval_s=60.0)
    if split_defender:
        defender_warmup.get()        # blocks only if the L4 is still cold-starting
        print("remote defender answered the warm-up call", flush=True)
    publish_policy(f"iter{start_iter} pre-rollout")
    # `until_iter` bounds THIS launch (exclusive), for budget-capped runs: a later launch resumes
    # from the checkpoint and carries on. It is an absolute iteration, not a count, because a
    # Modal retry after preemption re-enters with the same args and a resumed start_iter -- a
    # count would buy an extra iteration; an absolute bound stops at the same place either way.
    end_iter = min(N_ITERS, until_iter)
    if start_iter >= end_iter:
        print(f"nothing to do: resumed at iter {start_iter}, until_iter={until_iter}", flush=True)
        return
    # One structured row per iteration for the dashboard (scripts/watch_metrics.py). Additive:
    # the print lines below are unchanged, and a logging failure never stops the run.
    metrics_log = MetricsLog(os.path.join(ckpt_root, "metrics.jsonl"))
    for it in range(start_iter, end_iter):
        cap.buffer.clear()
        model.eval()
        rollout_dir = os.path.join(ckpt_root, f"rollout-iter{it}")
        plans = plans_for(it, rollout_dir)
        print(f"iter {it:3d} alloc    " + "  ".join(
            f"{p.kind[0]}:{p.task_id}" + (f"@{p.restart_turn}({p.reason[8:]})" if p.entry else "")
            for p in plans), flush=True)
        rollout_t0 = time.monotonic()
        examples = rollout(tasks, attack_agent_factory, benign_agent_factory,
                           defender_factory_for(it),
                           LocalShellSandbox, cap, n_rollouts=n_rollouts, max_turns=MAX_TURNS,
                           redaction_enforcement=REDACTION_ENFORCEMENT,
                           trained_side="attacker",
                           post_completion_turns=POST_COMPLETION_TURNS,
                           max_workers=ROLLOUT_WORKERS,
                           episode_store=rollout_dir,
                           commit=runs.commit, task_transform=episode_task_for(it),
                           reward_transform=reward_transform,
                           group_plans=plans, task_by_id=task_by_id, record_restart_state=True)
        # v6 dynamic sampling: dead base groups get at most alloc.max_replacements extra groups.
        # The plan is a function of the completed base episodes and is persisted, so a resume
        # replays the same replacements against the banked episodes instead of recomputing.
        # v7 same-iteration banking: this base pass's wins and near misses enter the banks BEFORE
        # replacements are planned, so a dead group can restart from a close call made by the policy
        # about to be trained. Base slots only (a resume may already hold replacement episodes);
        # finalize re-banks the same entries idempotently.
        bank_episodes({i: ep for i, ep in read_episodes(rollout_dir).items() if ep["slot"] < len(plans)},
                      banks, it, bank_record_dir, params=alloc)
        replacements = load_or_plan(
            os.path.join(rollout_dir, "replacements.json"),
            lambda: plan_replacements(read_episodes(rollout_dir), plans, banks, restart_depths,
                                      n_slots, n_rollouts,
                                      random.Random(f"replace-{LINEAGE}-{it}"), alloc,
                                      graduated=graduated, iteration=it))
        print(f"iter {it:3d} replace  " + ("  ".join(
            f"{p.slot}:{p.task_id}@{p.restart_turn}({p.reason})" for p in replacements) or "none"),
            flush=True)
        if replacements:
            examples += rollout(tasks, attack_agent_factory, benign_agent_factory,
                                defender_factory_for(it),
                                LocalShellSandbox, cap, n_rollouts=n_rollouts, max_turns=MAX_TURNS,
                                redaction_enforcement=REDACTION_ENFORCEMENT,
                                trained_side="attacker",
                                post_completion_turns=POST_COMPLETION_TURNS,
                                max_workers=ROLLOUT_WORKERS,
                                episode_store=rollout_dir,
                                commit=runs.commit, task_transform=episode_task_for(it),
                                reward_transform=reward_transform,
                                group_plans=replacements, task_by_id=task_by_id,
                                record_restart_state=True)
        rollout_min = (time.monotonic() - rollout_t0) / 60
        # Stats, banks and restart depths update from the episode store right away. All are safe to
        # repeat after a crash: stats and depths reload from the last checkpoint's meta, banking is
        # idempotent, and both plans are fixed by plans.json / replacements.json.
        summary, restart_depths, graduated = finalize_v7(
            read_episodes(rollout_dir), plans, replacements, banks, task_stats, restart_depths,
            graduated, it, n_rollouts, bank_record_dir, alloc)
        runs.commit()
        assign_advantages(examples, normalize_std=ADVANTAGE_STD_NORMALIZE)
        iter_gen_tokens = sum(len(ex.completion_ids) for ex in examples)
        cumulative_gen_tokens += iter_gen_tokens
        div = group_diversity([
            {"task_id": ex.group_key or ex.task_id, "episode_id": ex.episode_id,
             "reward": ex.reward,
             "verdicts": getattr(ex, "verdicts", [])}
            for ex in examples
        ])
        rew = reward_summary(examples)
        live = sum(1 for ex in examples if ex.advantage != 0)
        # win_rate is FRESH-START episodes only: restart groups begin from a favourable state, so
        # their rate is reported separately and is never the headline.
        win_rate = summary["fresh_win_rate"]
        print(f"iter {it:3d} rollout  live={live}/{len(examples)}  win_rate={win_rate:.0%}  "
              f"dead_groups={div['dead_group_rate']:.0%} "
              f"mixed_reward={div['mixed_reward_group_rate']:.0%}  |  "
              # The scalar the policy is optimizing. Component rates trade against each other;
              # a flat mean with anti-correlated components means cycling, not learning.
              f"mean_reward={rew['mean_reward']:+.3f} "
              f"(atk={rew['mean_reward_attack']:+.3f} ben={rew['mean_reward_benign']:+.3f})",
              flush=True)
        print(f"iter {it:3d} v5       fresh_groups={summary['n_fresh_groups']} "
              f"restart_groups={summary['n_restart_groups']} fallbacks={summary['fallbacks']}  "
              f"fresh_win={summary['fresh_win_rate']:.1%} (task-avg "
              f"{summary['fresh_win_rate_task_avg']:.1%})  restart_win={summary['restart_win_rate']:.1%}"
              f"  banked: wins={summary['banked_wins']} near={summary['banked_near']} "
              f"evicted={summary['evicted']}  bank sizes="
              f"{sum(len(b) for b in banks.values())} win routes="
              f"{sum(b.distinct_routes({3}) for b in banks.values())}", flush=True)
        depth_mean = summary["depth_frac_mean"]
        print(f"iter {it:3d} v6       dead_before={summary['dead_before']}/{summary['n_base_groups']} "
              f"dead_after={summary['dead_after']}  replacements={summary['replacements_live']}/"
              f"{summary['replacements']} live  depth_mean="
              f"{'-' if depth_mean is None else f'{depth_mean:.2f}'}  "
              f"win_by_depth={summary['win_by_depth']}  backoff={summary['backed_off']}", flush=True)
        print(f"iter {it:3d} v7       graduated={summary['graduated']} "
              f"(total {summary['graduated_total']})  same_iter_replacements="
              f"{summary['same_iter_replacements_live']}/{summary['same_iter_replacements']} live",
              flush=True)
        print(f"iter {it:3d} tokens   gen={iter_gen_tokens}  cum_gen={cumulative_gen_tokens}",
              flush=True)
        print(f"iter {it:3d} tactics  {tactic_summary(examples)}", flush=True)
        if skip_final_update and it == end_iter - 1:
            # Measurement / budget launch: the rollout is banked episode-by-episode, so the next
            # launch replays it from the bank for free and runs this iteration's update then.
            runs.commit()
            print(f"stopped after iter {it} rollout (--skip-final-update); its episodes are "
                  f"banked, relaunch to run the update and continue", flush=True)
            return

        model.train()
        torch.cuda.reset_peak_memory_stats()
        update_t0 = time.monotonic()
        for group in opt.param_groups:     # the KL-adaptive LR (checkpoint state), not the initial
            group["lr"] = current_lr
        m = update_step(learner, examples, beta=BETA, clip_eps=CLIP_EPS,
                        inner_epochs=INNER_EPOCHS,
                        no_checkpoint_max_tokens=NO_CHECKPOINT_MAX_TOKENS,
                        minibatches=MINIBATCHES, target_kl=lr_rule.target_kl,
                        shuffle_seed=f"{LINEAGE}-{it}", warmup_steps=WARMUP_STEPS,
                        global_step=optimizer_steps)
        optimizer_steps += m["steps"]
        m["lr"] = current_lr
        current_lr = next_lr(current_lr, m["kl_iter"], m["early_stopped"], lr_rule,
                             steps=m["steps"])
        update_min = (time.monotonic() - update_t0) / 60
        peak_gib = torch.cuda.max_memory_allocated() / 2**30   # before save/eval can move it
        # Same fields train_defender reports. grad_norm and epoch_ratios are what distinguish
        # "the gradient is tiny, raise LR" from "the gradient is fine, we need more steps" --
        # ratio alone cannot, and this script was silently dropping both.
        print(f"iter {it:3d} update   loss={m['loss']:.4f}  ratio={m['mean_ratio']:.3f}  "
              f"kl={m['mean_kl']:.5f}  grad={m['grad_norm']:.4f}  "
              f"epoch_ratios={[round(r, 4) for r in m['epoch_ratios']]}  "
              f"live={m['n_live_examples']}/{m['n_examples']}  "
              f"time={update_min:.1f}min  peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
              flush=True)
        print(f"iter {it:3d} steps    steps={m['steps']}  kl_iter={m['kl_iter']:.5f} "
              f"(target {lr_rule.target_kl})  clip_frac={m['clip_frac']:.3f}  "
              f"brake={'FIRED' if m['early_stopped'] else 'no'}  lr={m['lr']:.2e} -> {current_lr:.2e}",
              flush=True)

        ckpt = os.path.join(ckpt_root, f"iter{it}")
        model.save_pretrained(ckpt)
        torch.save(opt.state_dict(), os.path.join(ckpt, "optimizer.pt"))
        write_meta(ckpt, {"iter": it, "phase_identity": phase_identity,
                          "cumulative_gen_tokens": cumulative_gen_tokens,
                          "task_stats": [s.to_dict() for s in task_stats.values()],
                          "restart_depths": restart_depths,
                          "graduated": sorted(graduated), "lr": current_lr,
                          "optimizer_steps": optimizer_steps})
        publish_policy(f"iter{it} post-update")
        # Eval every HELDOUT_EVAL_EVERY iters (+ the final one). Single points are noisy (SE ~3pp
        # at n=128); only the trailing-window mean is read (spec s13).
        heldout_m = None
        # Overfit mode reads its one task's training win rate; the held-out set is not the question.
        if not overfit_task and ((it + 1) % HELDOUT_EVAL_EVERY == 0 or it == N_ITERS - 1
                or (eval_at_end and it == end_iter - 1)):
            heldout = _run_heldout_eval(HELDOUT_EVAL_EPISODES)
            win_rate = sum(r.outcome == "leak" for r in heldout) / len(heldout)
            stalled = mean_stalled_turn_rate([r.steps for r in heldout])
            # dead_rate is the probe-mix diagnostic: a stalled gap across arms is only trustworthy
            # when dead_rate is comparable across arms (spec s13 review point 2).
            dead = sum(dead_observation_rate(r.steps) for r in heldout) / len(heldout)
            print(f"iter {it:3d} heldout  n={len(heldout)}  win_rate={win_rate:.1%}  "
                  f"stalled_rate={stalled:.1%}  dead_rate={dead:.1%}", flush=True)
            heldout_m = {"win_rate": win_rate, "stalled_rate": stalled, "dead_rate": dead}
            try:   # per-seed record for paired comparisons; telemetry, never fatal
                save_record(os.path.join(ckpt_root, "heldout", f"iter{it}.json"),
                            make_record(_heldout_seeds(len(heldout)),
                                        [r.outcome for r in heldout],
                                        turns=[r.n_turns for r in heldout], policy=ckpt_root,
                                        iter=it, cum_gen=cumulative_gen_tokens,
                                        stalled_rate=stalled, dead_rate=dead))
            except Exception as exc:
                print(f"[heldout] iter {it}: could not save the per-seed record: {exc!r}",
                      flush=True)
        try:   # telemetry must never cost the run: a builder KeyError is a warning, not a crash
            stall = training_stall(read_episodes(rollout_dir))
            fmt = lambda x: "-" if x is None else f"{x:.1%}"  # noqa: E731
            print(f"iter {it:3d} stall    wins={fmt(stall['learning.stalled_win'])}  "
                  f"losses={fmt(stall['learning.stalled_loss'])}  (verbatim-repeated observations)",
                  flush=True)
            metrics = attacker_metrics(
                live=live, n_examples=len(examples), div=div, rew=rew, summary=summary,
                tactics=tactic_counts(examples), update=m, rollout_min=rollout_min,
                update_min=update_min, gen_tokens=iter_gen_tokens, peak_gib=peak_gib,
                banks_size=sum(len(b) for b in banks.values()),
                win_routes=sum(b.distinct_routes({3}) for b in banks.values()),
                heldout=heldout_m, stall=stall)
            metrics_log.log(it, "attacker", cum_gen=cumulative_gen_tokens, **metrics)
        except Exception as exc:
            print(f"[metrics] iter {it}: could not build metrics: {exc!r}", flush=True)
        prune_checkpoints(ckpt_root, keep=CKPT_KEEP)
        # The rollout bank is only disposable once THIS iteration's checkpoint exists: dropping it
        # earlier would make a preemption between update and checkpoint re-run the whole rollout.
        shutil.rmtree(os.path.join(ckpt_root, f"rollout-iter{it}"), ignore_errors=True)
        runs.commit()

    if end_iter < N_ITERS:
        print(f"stopped at --until-iter {until_iter}; relaunch to resume at iter {end_iter}",
              flush=True)
        return
    model.save_pretrained(f"{ckpt_root}/final")
    runs.commit()


# cpu=8: without it the container gets Modal's minimal default -- `nproc` measured 1 (2026-10-05).
# Two vLLM engine loops, two API servers, 48 rollout threads + their shell sandboxes and the trainer
# all shared one core, so the attacker engine decoded at a FIXED ~300ms/step whether 13 or 47 seqs
# were running (GPU waiting on the CPU: 70-100% "util" at only ~230W). CPU is billed per core and
# is cheap next to the A100 it was starving.
@app.function(gpu="A100-80GB", cpu=8.0, timeout=24 * 60 * 60,
              retries=modal.Retries(initial_delay=0.0, max_retries=10),
              volumes={"/cache/huggingface": hf_cache, "/runs": runs})
def train(arm: int = 3, repeat_penalty_coef: float = 0.02, until_iter: int = N_ITERS,
          skip_final_update: bool = False, split_defender: bool = False,
          eval_at_end: bool = False, n_rollouts: int = N_ROLLOUTS, baseline_eval: bool = False,
          max_turns: int = MAX_TURNS, overfit_task: str = ""):
    """Own the vLLM lifecycle OUTSIDE the training body.

    Modal can retry a failed input in the SAME container. Without this, the previous attempt's
    servers stay alive holding ~52 GiB and the retry OOMs before it can even take a CUDA context
    -- which is exactly what happened. Every return and every exception must release them.

    A trainer OOM is RETURNED, not raised. `retries` exists for preemption, but Modal retries any
    exception, and an OOM on the same banked rollout is deterministic: 2026-10-18's v4 OOM would
    have rebuilt both engines (~7 min each) ten times to fail identically. Returning ends the run;
    the bank and every checkpoint survive, so the relaunch after a memory fix resumes in place.
    """
    import torch
    active_servers = []
    try:
        return _train_impl(active_servers, arm, repeat_penalty_coef, until_iter,
                           skip_final_update, split_defender, eval_at_end, n_rollouts,
                           baseline_eval, max_turns, overfit_task)
    except torch.OutOfMemoryError as e:
        print(f"TRAINER OOM -- NOT retrying (deterministic on a banked rollout): {e}", flush=True)
        return "oom"
    finally:
        for stop_server, proc in reversed(active_servers):
            stop_server(proc)


@app.local_entrypoint()
def main(arm: int = 3, repeat_penalty_coef: float = 0.02, until_iter: int = N_ITERS,
         skip_final_update: bool = False, split_defender: bool = False,
         eval_at_end: bool = False, n_rollouts: int = N_ROLLOUTS, baseline_eval: bool = False,
          max_turns: int = MAX_TURNS, overfit_task: str = ""):
    if train.remote(arm, repeat_penalty_coef, until_iter, skip_final_update, split_defender,
                    eval_at_end, n_rollouts, baseline_eval, max_turns, overfit_task) == "oom":
        print("run ended on a trainer OOM (not retried) -- see the remote log for the allocation")
