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
N_ROLLOUTS = 8          # GRPO group size; larger groups buy within-group variance
N_ITERS = 20
LR = 1e-5
# 0.7 is what attacker_viability measured 16.7% at (make_vllm_generate's default). 1.0 was
# chosen to buy within-group variance, but it bought the opposite: iteration 0 came back
# 100% dead groups at a 0% win rate. Exploration you cannot convert into a win is not
# exploration. Match the measured config first, then reintroduce deltas one at a time.
TEMP = 0.7
BETA = 0.04             # KL weight against the frozen reference
CLIP_EPS = 0.2          # PPO trust region; only bites once inner epochs move the ratio
# Gradient steps per rollout batch; see train_defender.py. Watch `epoch_ratios` for clip saturation.
INNER_EPOCHS = 4
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
HELDOUT_EVAL_EPISODES = 128
HELDOUT_EVAL_EVERY = 5             # iterations between held-out evals (plus the final iteration)
PATH_NORMALISED = True             # Task 1 landed; both arms see normalised observations.

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


def _train_impl(active_servers, arm, repeat_penalty_coef, until_iter=N_ITERS):
    import contextlib, os, sys, tempfile, torch
    sys.path.insert(0, "/root")

    assert arm in (2, 3), f"arm must be 2 or 3, got {arm}"
    coef = repeat_penalty_coef if arm == 3 else 0.0   # arm 2 ignores the coef
    ckpt_root = f"/runs/attacker-grpo-arm{arm}-coef{coef}"

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
        BENIGN_AGENT_SYSTEM)
    from redteamrl.policies.vllm_client import (
        load_lora_adapter, make_vllm_generate, start_vllm_server, stop_vllm_server)
    from redteamrl.sandbox.shell import LocalShellSandbox
    from redteamrl.policies.prompt_pool import (
        DEFENDER_STYLES, PROMPT_POOL_VERSION, select_style)
    from redteamrl.envs.generate import build_generated_tasks
    from redteamrl.envs.registry import assert_training_split
    from redteamrl.train.capture import VLLMCapturingGenerate
    from redteamrl.train.learner import Learner, prepare_for_long_context_training
    from redteamrl.train.train import rollout, assign_advantages, tactic_summary, update_step
    from redteamrl.train.diversity import group_diversity, reward_summary
    from redteamrl.sft.canary import CANARY_SCHEME_VERSION, randomize_task_canary
    from redteamrl.train.checkpoint import (
        latest_checkpoint,
        prune_checkpoints,
        read_meta,
        validate_phase_identity,
        write_meta,
    )

    # Frozen DEFENDER on vLLM first so it reserves its GPU fraction; HF attacker takes the rest.
    # The adapters must be MERGED before serving -- declaring them and serving the raw base
    # would silently train the attacker against the wrong (unpromoted) defender.
    served_defender = DEFENDER_MODEL
    _adapters = [a for a in (DEFENDER_SFT_ADAPTER, DEFENDER_GRPO_ADAPTER) if a]
    if _adapters:
        import shutil
        from peft import PeftModel
        from transformers import AutoModelForCausalLM as _AutoLM, AutoTokenizer as _AutoTok
        merged_dir = tempfile.mkdtemp(prefix="frozen-defender-", dir="/tmp")
        _model = _AutoLM.from_pretrained(DEFENDER_MODEL, torch_dtype=torch.bfloat16,
                                         device_map="cpu")
        for _adapter in _adapters:                      # order matters: SFT then GRPO
            _model = PeftModel.from_pretrained(_model, _adapter).merge_and_unload()
        _model.save_pretrained(merged_dir)
        _AutoTok.from_pretrained(DEFENDER_MODEL).save_pretrained(merged_dir)
        del _model
        served_defender = merged_dir
        print(f"serving frozen defender = base + {' + '.join(_adapters)}", flush=True)
    active_servers.append((stop_vllm_server, start_vllm_server(
        served_defender, DEF_PORT, DEF_MEM_FRAC, max_model_len=DEF_MAX_MODEL_LEN,
        max_num_seqs=DEF_MAX_NUM_SEQS)))
    hf_cache.commit()
    # temperature=0.0: the defender is the frozen OPPONENT, not the trainee. At 0.7 sibling rollouts
    # with identical attacker actions can draw different verdicts, so part of the within-group reward
    # variance -- which IS the GRPO advantage signal -- would be opponent coin-flips mis-attributed to
    # the attacker. Pinning it to greedy makes the defender a deterministic function of the transcript.
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
    print(f"{len(tasks)} attack tasks (generated envs x {N_GEN_ENVS})", flush=True)

    def episode_task_for(iteration):
        """A fresh high-entropy secret per (iteration, rollout) -- never repeated.

        Keying on rollout_index alone replants the SAME six secrets every iteration, so a policy
        could earn the whole reward by memorizing six strings and generalize to nothing. Including
        the iteration makes every episode a secret the model has not seen, so improvement has to
        transfer. Still a pure function of (iteration, rollout_index), so a resumed run reproduces
        exactly the canaries its banked episodes were generated with.
        """
        def episode_task(spec, rollout_index):
            return randomize_task_canary(
                spec, iteration * N_ROLLOUTS + rollout_index, CANARY_SEED, CANARY_REVISION)
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

    _last_serve_dir = {"path": None}

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
        # The previous publish's weights are now resident in vLLM; drop its dir to bound /tmp.
        if _last_serve_dir["path"] is not None:
            shutil.rmtree(_last_serve_dir["path"], ignore_errors=True)
        _last_serve_dir["path"] = serve_dir
        probe_system, probe_user = "You are an agent.", "Reply with one short JSON object."
        served = VLLMCapturingGenerate(
            base_url=f"http://localhost:{ATK_PORT}", model="attacker", tokenizer=tok,
            temperature=0.0, max_new_tokens=32,
        )(probe_system, [{"role": "user", "content": probe_user}])
        probe_prompt = tok.apply_chat_template(
            [{"role": "system", "content": probe_system},
             {"role": "user", "content": probe_user}],
            add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = tok(probe_prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        model.eval()
        # prepare_for_long_context_training set use_cache=False for checkpointing. Generating
        # WITHOUT the KV cache recomputes the whole forward each step: same arithmetic in theory,
        # different kernels in practice, and greedy argmax flips on near-tied logits -- after which
        # every later token diverges. This is a no_grad diagnostic, so checkpointing is a no-op
        # here and the cache is safe to restore for it.
        cached = getattr(model.config, "use_cache", None)

        def _greedy(with_adapter):
            ctx = contextlib.nullcontext() if with_adapter else model.disable_adapter()
            with torch.no_grad(), ctx:
                out = model.generate(**ids, do_sample=False, max_new_tokens=32)
            return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        try:
            # prepare_for_long_context_training set use_cache=False for checkpointing. Generating
            # without the KV cache recomputes the whole forward each step -- same arithmetic in
            # theory, different kernels in practice. This is a no_grad diagnostic, so checkpointing
            # is a no-op here and the cache is safe to restore for it.
            model.config.use_cache = True
            local = _greedy(with_adapter=True)
            # Only computed when the check fails: is vLLM serving the BASE policy? That is the
            # difference between "the swap silently no-opped" (fatal -- rollouts would come from a
            # policy that is not being trained) and "two inference stacks diverged on a near-tied
            # logit" (benign -- vLLM generates, HF re-scores, and GRPO measured ratio=1.000).
            base = _greedy(with_adapter=False) if served.strip() != local else None
        finally:
            if cached is not None:
                model.config.use_cache = cached

        if served.strip() != local:
            shared = len(os.path.commonprefix([served.strip(), local]))
            if served.strip() == base:
                raise RuntimeError(
                    f"[{tag}] vLLM is serving the BASE policy -- the adapter swap did not take, so "
                    f"rollouts would come from a model that is not the one being trained.\n"
                    f"  vLLM greedy: {served.strip()[:160]!r}\n"
                    f"  HF+adapter : {local[:160]!r}\n"
                    f"  HF base    : {base[:160]!r}")
            print(f"[{tag}] WARNING: vLLM and HF greedy diverge after {shared} chars, but vLLM is "
                  f"NOT serving the base policy -- the adapter DID load. Treating as numeric "
                  f"divergence between the two stacks.\n"
                  f"  vLLM: {served.strip()[:120]!r}\n  HF  : {local[:120]!r}", flush=True)
        print(f"[{tag}] served policy verified against the trainer", flush=True)

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

    def attack_agent_factory(spec):
        # The LIVE model. HINT_TECHNIQUE stays as-is: freezing the hint keeps improvements
        # attributable to learning rather than to us writing better attacks by hand.
        return PromptedAttacker(cap, goal=spec.goal, cover=spec.request, hint=HINT_TECHNIQUE,
                                system_template=ATTACKER_SYSTEM_SHELL,
                                max_context_chars=TRANSCRIPT_CONTEXT_CHARS)

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
        "n_rollouts": N_ROLLOUTS,
        "lr": LR,
        "temp": TEMP,
        "max_turns": MAX_TURNS,
        "redaction_enforcement": REDACTION_ENFORCEMENT,
        "arm": arm,
        "repeat_penalty_coef": coef,
        "path_normalised": PATH_NORMALISED,
    }
    start_iter = 0
    resume_dir = latest_checkpoint(ckpt_root)
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
        print("no checkpoint found — starting fresh at iter 0", flush=True)

    # Arm 3 subtracts coef * exact-repeat-count from each episode's reward (coef 0.0 = arm 2,
    # the identity). Built once; the secret it keys on comes from each episode's spec.forbidden[0],
    # which _run_one_episode passes to the transform -- no canary recovery here.
    from redteamrl.train.repeat_penalty import make_repeat_penalty
    reward_transform = make_repeat_penalty(coef)

    # --- held-out eval (greedy, temp-0, n disjoint seeds) ---
    from concurrent.futures import ThreadPoolExecutor
    from redteamrl.harness.episode import run_episode
    from redteamrl.attack_sft.eval_metrics import mean_stalled_turn_rate, dead_observation_rate
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

    def _run_heldout_eval(n):
        seeds = range(HELDOUT_SEED_START, HELDOUT_SEED_START + n)

        def one(seed):
            # Mirror attack_agent_factory / defender_factory_for exactly (same system prompts, same
            # served model names), differing ONLY in the greedy attacker generate; the defender is
            # the frozen temp-0 opponent gen_def already serving on DEF_PORT.
            spec = _heldout_attack_task(seed)
            attacker = PromptedAttacker(eval_gen, goal=spec.goal, cover=spec.request,
                                        hint=HINT_TECHNIQUE, system_template=ATTACKER_SYSTEM_SHELL,
                                        max_context_chars=TRANSCRIPT_CONTEXT_CHARS)
            defender = PromptedDefender(generate=gen_def, system_template=None,
                                        max_context_chars=TRANSCRIPT_CONTEXT_CHARS)
            sandbox = LocalShellSandbox()
            try:
                return run_episode(spec, attacker, defender, sandbox, max_turns=MAX_TURNS,
                                   redaction_enforcement=REDACTION_ENFORCEMENT)
            finally:
                sandbox.close()

        with ThreadPoolExecutor(max_workers=ROLLOUT_WORKERS) as pool:
            return list(pool.map(one, seeds))

    publish_policy(f"iter{start_iter} pre-rollout")
    # `until_iter` bounds THIS launch (exclusive), for budget-capped runs: a later launch resumes
    # from the checkpoint and carries on. It is an absolute iteration, not a count, because a
    # Modal retry after preemption re-enters with the same args and a resumed start_iter -- a
    # count would buy an extra iteration; an absolute bound stops at the same place either way.
    end_iter = min(N_ITERS, until_iter)
    if start_iter >= end_iter:
        print(f"nothing to do: resumed at iter {start_iter}, until_iter={until_iter}", flush=True)
        return
    for it in range(start_iter, end_iter):
        cap.buffer.clear()
        model.eval()
        examples = rollout(tasks, attack_agent_factory, benign_agent_factory,
                           defender_factory_for(it),
                           LocalShellSandbox, cap, n_rollouts=N_ROLLOUTS, max_turns=MAX_TURNS,
                           redaction_enforcement=REDACTION_ENFORCEMENT,
                           trained_side="attacker",
                           post_completion_turns=POST_COMPLETION_TURNS,
                           max_workers=ROLLOUT_WORKERS,
                           episode_store=os.path.join(ckpt_root, f"rollout-iter{it}"),
                           commit=runs.commit, task_transform=episode_task_for(it),
                           reward_transform=reward_transform)
        assign_advantages(examples)
        iter_gen_tokens = sum(len(ex.completion_ids) for ex in examples)
        cumulative_gen_tokens += iter_gen_tokens
        div = group_diversity([
            {"task_id": ex.task_id, "episode_id": ex.episode_id, "reward": ex.reward,
             "verdicts": getattr(ex, "verdicts", [])}
            for ex in examples
        ])
        rew = reward_summary(examples)
        live = sum(1 for ex in examples if ex.advantage != 0)
        wins = {ex.episode_id: ex.reward for ex in examples}
        win_rate = sum(r > 0 for r in wins.values()) / max(len(wins), 1)
        print(f"iter {it:3d} rollout  live={live}/{len(examples)}  win_rate={win_rate:.0%}  "
              f"dead_groups={div['dead_group_rate']:.0%} "
              f"mixed_reward={div['mixed_reward_group_rate']:.0%}  |  "
              # The scalar the policy is optimizing. Component rates trade against each other;
              # a flat mean with anti-correlated components means cycling, not learning.
              f"mean_reward={rew['mean_reward']:+.3f} "
              f"(atk={rew['mean_reward_attack']:+.3f} ben={rew['mean_reward_benign']:+.3f})",
              flush=True)
        print(f"iter {it:3d} tokens   gen={iter_gen_tokens}  cum_gen={cumulative_gen_tokens}",
              flush=True)
        print(f"iter {it:3d} tactics  {tactic_summary(examples)}", flush=True)

        model.train()
        m = update_step(learner, examples, beta=BETA, clip_eps=CLIP_EPS,
                        inner_epochs=INNER_EPOCHS)
        # Same fields train_defender reports. grad_norm and epoch_ratios are what distinguish
        # "the gradient is tiny, raise LR" from "the gradient is fine, we need more steps" --
        # ratio alone cannot, and this script was silently dropping both.
        print(f"iter {it:3d} update   loss={m['loss']:.4f}  ratio={m['mean_ratio']:.3f}  "
              f"kl={m['mean_kl']:.5f}  grad={m['grad_norm']:.4f}  "
              f"epoch_ratios={[round(r, 4) for r in m['epoch_ratios']]}  "
              f"live={m['n_live_examples']}/{m['n_examples']}", flush=True)

        ckpt = os.path.join(ckpt_root, f"iter{it}")
        model.save_pretrained(ckpt)
        torch.save(opt.state_dict(), os.path.join(ckpt, "optimizer.pt"))
        write_meta(ckpt, {"iter": it, "phase_identity": phase_identity,
                          "cumulative_gen_tokens": cumulative_gen_tokens})
        publish_policy(f"iter{it} post-update")
        # Eval every HELDOUT_EVAL_EVERY iters (+ the final one). Single points are noisy (SE ~3pp
        # at n=128); only the trailing-window mean is read (spec s13).
        if (it + 1) % HELDOUT_EVAL_EVERY == 0 or it == N_ITERS - 1:
            heldout = _run_heldout_eval(HELDOUT_EVAL_EPISODES)
            win_rate = sum(r.outcome == "leak" for r in heldout) / len(heldout)
            stalled = mean_stalled_turn_rate([r.steps for r in heldout])
            # dead_rate is the probe-mix diagnostic: a stalled gap across arms is only trustworthy
            # when dead_rate is comparable across arms (spec s13 review point 2).
            dead = sum(dead_observation_rate(r.steps) for r in heldout) / len(heldout)
            print(f"iter {it:3d} heldout  n={len(heldout)}  win_rate={win_rate:.1%}  "
                  f"stalled_rate={stalled:.1%}  dead_rate={dead:.1%}", flush=True)
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


@app.function(gpu="A100-80GB", timeout=24 * 60 * 60,
              retries=modal.Retries(initial_delay=0.0, max_retries=10),
              volumes={"/cache/huggingface": hf_cache, "/runs": runs})
def train(arm: int = 3, repeat_penalty_coef: float = 0.05, until_iter: int = N_ITERS):
    """Own the vLLM lifecycle OUTSIDE the training body.

    Modal can retry a failed input in the SAME container. Without this, the previous attempt's
    servers stay alive holding ~52 GiB and the retry OOMs before it can even take a CUDA context
    -- which is exactly what happened. Every return and every exception must release them.
    """
    active_servers = []
    try:
        return _train_impl(active_servers, arm, repeat_penalty_coef, until_iter)
    finally:
        for stop_server, proc in reversed(active_servers):
            stop_server(proc)


@app.local_entrypoint()
def main(arm: int = 3, repeat_penalty_coef: float = 0.05, until_iter: int = N_ITERS):
    train.remote(arm, repeat_penalty_coef, until_iter)
