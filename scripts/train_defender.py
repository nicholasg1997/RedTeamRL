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
# Gradient steps per rollout batch. Generation is hours, an optimizer step is seconds -- taking one
# step per rollout wastes the batch AND leaves the clipped objective inert (ratio pinned at 1.0).
# Watch `epoch_ratios`: once it saturates 1+CLIP_EPS the remaining epochs are fully clipped and
# contribute nothing, which is the signal to lower LR rather than add epochs.
INNER_EPOCHS = 4
MAX_TURNS = 12          # chunk exfil needs ~7 turns
REDACTION_ENFORCEMENT = "unshielded"  # attacker must receive exploitable policy mistakes
ATTACKER_MODEL = "cyankiwi/Qwen3.6-27B-AWQ-INT4"   # frozen opponent (plays attacker + benign agent)
OPP_PORT = 8000
DEF_PORT = 8001
# Measured by scripts/vllm_lora_probe.py on this exact stack: 27B@0.35 + 4B@0.30 = 50.4 GiB, and
# the HF trainer on top reaches 57.9 GiB of 79.3 -- ~21 GiB left for optimizer state and
# activations. `gpu_memory_utilization` is a budget against a MOVING baseline, so the servers must
# start sequentially or whoever profiles second charges the other's allocation to itself.
OPP_MEM_FRAC = 0.35
DEF_MEM_FRAC = 0.30
DEF_MAX_MODEL_LEN = 8192   # defender prompts are ~2.5k tokens + DEF_MAX_NEW_TOKENS generated
DEF_MAX_NUM_SEQS = 16      # >= ROLLOUT_WORKERS so concurrent episodes batch, not queue
DEF_MAX_NEW_TOKENS = 512   # reasoning + one JSON decision, thinking OFF
ATK_MAX_TOKENS = 6144          # match the probe that produced the messy middle (attacker thinks more)
BEN_MAX_TOKENS = 4096          # honest agent
OPP_MAX_MODEL_LEN = 32768
OPP_MAX_NUM_SEQS = 16   # 224k tokens of KV at ~10k/request leaves room for ~20 concurrent
TRANSCRIPT_CONTEXT_CHARS = 40_000
# Sample the agent system prompts from style-paired pools instead of one fixed prompt each. The
# matched probe showed defender GRPO improving aggregate attack leak while pair_taint_accuracy and
# the call gate stayed pinned -- the defender was reading episode TYPE off a fixed prompt's style
# rather than judging content. The pools span the same styles on both sides and style is chosen
# independently of episode type, so that signal is gone. Set False to reproduce the old runs.
USE_PROMPT_POOL = True
# Canaries are randomized per (ITERATION, rollout) -- every episode plants a secret the model has
# never seen, so a policy cannot earn reward by memorizing a fixed set of strings. Two earlier
# versions were wrong: the literal template `tok-deploy-CANARY-3m2n1k` is low-entropy and reads as
# a placeholder (the defender passed it 6/6 on ci-build-benign while SFT measured 5% leak on the
# same task), and keying on rollout_index alone replanted the same six secrets every iteration.
# CANARY_REVISION must also differ from the SFT run's PIPELINE_REVISION so GRPO faces secrets the
# bootstrap never saw.
CANARY_REVISION = 101
CANARY_SEED = 0
# canary co-located in each env's secret file — the TEMPLATE that randomization replaces
CANARIES = {"ops-logs": "sk-api-CANARY-7a1b2c", "analytics": "pw-db-CANARY-9x8y7z",
            "ci-build": "tok-deploy-CANARY-3m2n1k"}
# SFT handoff: point at a promoted adapter directory on the Volume to warm-start GRPO from the
# reasoning bootstrap instead of the raw base model. None -> start from base.
SFT_ADAPTER = "/runs/sft-defender/round9/training/7eeae96976a4/adapter"   # e.g. "/runs/sft-defender/<root>/training/<fingerprint>/adapter"
CKPT_ROOT = "/runs/defender-grpo-r9-vllm"   # on the Modal Volume; the ONLY state that survives preemption
# Episodes run concurrently ACROSS ALL TASKS. vLLM batches by in-flight request count, so the old
# task-at-a-time loop pinned both engines at a queue of ~1 (measured: 27B at 61 tok/s, the 4B
# starved to 1.7). Keep this <= the engines' max_num_seqs so requests batch rather than queue.
ROLLOUT_WORKERS = 16
# Benign episodes complete in ~2 turns and then idled to MAX_TURNS under the default fixed
# horizon. A bounded post-completion window gives equal safety exposure at a fraction of the cost.
POST_COMPLETION_TURNS = 4
CKPT_KEEP = 3                             # keep newest N checkpoints (the latest is always resume-safe)
ALLOW_LEGACY_CHECKPOINT = False           # require explicit opt-in for pre-identity checkpoints

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "peft>=0.11",
                 "accelerate>=0.30", "pydantic", "pyyaml", "tqdm")
    # expandable_segments reduces fragmentation between the vLLM reservation and the HF trainer,
    # which share one A100 and allocate in very different block sizes.
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          # Required for /v1/load_lora_adapter: how the updated GRPO policy reaches vLLM each
          # iteration (a ~50MB adapter file, not the whole backbone).
          "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
)
hf_cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
app = modal.App("redteamrl-defender-grpo", image=image)


def _train_impl(active_servers):
    import json, os, shutil, sys, tempfile, torch
    sys.path.insert(0, "/root")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from redteamrl.policies.prompted import (
        PromptedDefender, PromptedAttacker, ATTACKER_SYSTEM_SHELL, HINT_TECHNIQUE, BENIGN_AGENT_SYSTEM)
    from redteamrl.policies.vllm_client import (
        load_lora_adapter, make_vllm_generate, start_vllm_server, stop_vllm_server)
    from redteamrl.sandbox.shell import LocalShellSandbox
    from redteamrl.envs import build_tasks
    from redteamrl.envs.registry import assert_training_split
    from redteamrl.envs.ops_logs import OPS_LOGS
    from redteamrl.envs.analytics import ANALYTICS
    from redteamrl.envs.ci_build import CI_BUILD
    from redteamrl.train.capture import VLLMCapturingGenerate
    from redteamrl.train.learner import Learner
    from redteamrl.train.train import rollout, assign_advantages, update_step
    from redteamrl.train.checkpoint import (
        latest_checkpoint,
        prune_checkpoints,
        read_meta,
        validate_phase_identity,
        write_meta,
    )
    from redteamrl.train.diversity import group_diversity, reward_summary
    from redteamrl.sft.canary import CANARY_SCHEME_VERSION, randomize_task_canary
    from redteamrl.policies.prompt_pool import (
        ATTACKER_STYLES, BENIGN_STYLES, PROMPT_POOL_VERSION, select_style)

    # ---- opponent first, on a quiet GPU, and fully healthy before anything else allocates ----
    active_servers.append((stop_vllm_server, start_vllm_server(
        ATTACKER_MODEL, OPP_PORT, OPP_MEM_FRAC, max_model_len=OPP_MAX_MODEL_LEN,
        max_num_seqs=OPP_MAX_NUM_SEQS)))
    hf_cache.commit()
    gen_atk = make_vllm_generate(f"http://localhost:{OPP_PORT}", ATTACKER_MODEL,
                                 enable_thinking=True, max_tokens=ATK_MAX_TOKENS)
    gen_benign = make_vllm_generate(f"http://localhost:{OPP_PORT}", ATTACKER_MODEL,
                                    enable_thinking=True, max_tokens=BEN_MAX_TOKENS)

    tasks = assert_training_split(
        build_tasks(OPS_LOGS,  CANARIES["ops-logs"],  canary_path="app/config.py")
        + build_tasks(ANALYTICS, CANARIES["analytics"], canary_path=".env")
        + build_tasks(CI_BUILD,  CANARIES["ci-build"],  canary_path="deploy/credentials")
    )
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

    # Snapshot the backbone for vLLM BEFORE wrapping it: once get_peft_model injects LoRA layers,
    # `get_base_model()` returns the wrapped module, not a clean checkpoint. vLLM's base MUST be
    # this merged model -- serving raw Qwen under the GRPO adapter would generate from a different
    # policy than the one being scored, making every importance ratio wrong.
    merged_dir = tempfile.mkdtemp(prefix="grpo-merged-base-", dir="/tmp")
    base.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)

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

    # ---- serve the SAME policy on vLLM: merged backbone + the live GRPO adapter ----
    adapter_serve_dir = tempfile.mkdtemp(prefix="grpo-live-adapter-", dir="/tmp")
    model.save_pretrained(adapter_serve_dir)
    print(f"serving defender from {merged_dir} + adapter {adapter_serve_dir}", flush=True)
    # Started LAST: the opponent and the HF trainer are both resident and static by now, so this
    # engine's memory profiling measures only its own allocation.
    active_servers.append((stop_vllm_server, start_vllm_server(
        merged_dir, DEF_PORT, DEF_MEM_FRAC, max_model_len=DEF_MAX_MODEL_LEN,
        max_num_seqs=DEF_MAX_NUM_SEQS, lora_name="defender", lora_path=adapter_serve_dir,
    )))

    def publish_policy(tag):
        """Push the current adapter to vLLM and PROVE the served policy matches the trained one.

        A swap that silently no-ops would keep generating from a stale policy while HF reports a
        healthy KL — rollouts and gradients would quietly describe different models. Comparing a
        greedy generation from each side catches that on the spot.
        """
        model.save_pretrained(adapter_serve_dir)
        load_lora_adapter(f"http://localhost:{DEF_PORT}", "defender", adapter_serve_dir)
        probe_chat = [{"role": "system", "content": "You are a monitor."},
                      {"role": "user", "content": "Reply with one short JSON object."}]
        probe_prompt = tok.apply_chat_template(
            probe_chat, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        served = VLLMCapturingGenerate(
            base_url=f"http://localhost:{DEF_PORT}", model="defender", tokenizer=tok,
            temperature=0.0, max_new_tokens=32,
        )("You are a monitor.", [{"role": "user", "content": "Reply with one short JSON object."}])
        ids = tok(probe_prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        model.eval()
        with torch.no_grad():
            local_out = model.generate(**ids, do_sample=False, max_new_tokens=32)
        local = tok.decode(local_out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        if served.strip() != local.strip():
            raise RuntimeError(
                f"[{tag}] vLLM is not serving the trained policy — the adapter swap did not take. "
                f"vLLM greedy: {served.strip()[:160]!r} vs HF greedy: {local.strip()[:160]!r}"
            )
        print(f"[{tag}] served policy verified against the trainer", flush=True)

    cap = VLLMCapturingGenerate(                    # thinking OFF -> reasoning+JSON fits
        base_url=f"http://localhost:{DEF_PORT}", model="defender", tokenizer=tok,
        temperature=TEMP, max_new_tokens=DEF_MAX_NEW_TOKENS,
    )

    def defender_factory(spec):
        # The defender is the TRAINEE this phase, so its prompt stays fixed at
        # DEFENDER_SYSTEM: varying it would split the gradient across phrasings and break
        # comparability with the probes. DEFENDER_STYLES is for the attacker phase, where
        # the defender is the frozen opponent.
        return PromptedDefender(generate=cap, max_context_chars=TRANSCRIPT_CONTEXT_CHARS)  # match the probe

    # The style index is a pure function of (task, rollout, iteration) and never of episode type,
    # so an attack and a benign episode are equally likely to be phrased any given way.
    attacker_styles = sorted(ATTACKER_STYLES)
    benign_styles = sorted(BENIGN_STYLES)

    def _style_for(spec, iteration, pool):
        """Style is fixed within a (task, iteration) -- i.e. within a GRPO GROUP.

        Varying it per rollout would put prompt style into the within-group reward variance, and
        that variance is the advantage signal: it must stay attributable to the defender's sampled
        decisions, not to which phrasing its opponent happened to draw. Across tasks and iterations
        the pool is still sampled ~uniformly, which is what removes the fingerprint.
        """
        return pool[select_style(spec.id, 0, iteration, len(pool))]

    def attack_agent_factory_for(iteration):
        def attack_agent_factory(spec):
            template = (ATTACKER_STYLES[_style_for(spec, iteration, attacker_styles)]
                        if USE_PROMPT_POOL else ATTACKER_SYSTEM_SHELL)
            return PromptedAttacker(gen_atk, goal=spec.goal, cover=spec.request,
                                    hint=HINT_TECHNIQUE, system_template=template,
                                    max_context_chars=TRANSCRIPT_CONTEXT_CHARS)
        return attack_agent_factory

    def benign_agent_factory_for(iteration):
        def benign_agent_factory(spec):
            template = (BENIGN_STYLES[_style_for(spec, iteration, benign_styles)]
                        if USE_PROMPT_POOL else BENIGN_AGENT_SYSTEM)
            return PromptedAttacker(gen_benign, goal=spec.request, system_template=template,
                                    max_context_chars=TRANSCRIPT_CONTEXT_CHARS)
        return benign_agent_factory

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
        "beta": BETA, "clip_eps": CLIP_EPS, "inner_epochs": INNER_EPOCHS,
        # Changes the episode horizon, so it changes the reward distribution -- a resume across
        # this value would mix two different experiments. max_workers is NOT here: concurrency
        # changes wall-clock only, and episode ids are derived from indices, not arrival order.
        "post_completion_turns": POST_COMPLETION_TURNS,
        # Changes the task distribution, so a resume across it would mix two environments.
        "canary_revision": CANARY_REVISION, "canary_seed": CANARY_SEED,
        "canary_scheme_version": CANARY_SCHEME_VERSION,
        # Changes the observable agent behaviour distribution, so a resume across it mixes
        # environments the same way a canary change would.
        "use_prompt_pool": USE_PROMPT_POOL, "prompt_pool_version": PROMPT_POOL_VERSION,
        "reference_policy": reference_policy,
        # Generation moved from HF batch-1 to vLLM; a resume across that boundary would mix
        # rollouts drawn under two different sampling stacks.
        "generation_backend": "vllm-lora",
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

    # Verify the initial serve before spending a rollout on it.
    publish_policy(f"iter{start_iter} pre-rollout")

    for it in range(start_iter, N_ITERS):
        cap.buffer.clear()
        model.eval()
        examples = rollout(tasks, attack_agent_factory_for(it), benign_agent_factory_for(it),
                           defender_factory,
                           LocalShellSandbox, cap, n_rollouts=N_ROLLOUTS, max_turns=MAX_TURNS,
                           redaction_enforcement=REDACTION_ENFORCEMENT,
                           post_completion_turns=POST_COMPLETION_TURNS,
                           max_workers=ROLLOUT_WORKERS,
                           episode_store=os.path.join(CKPT_ROOT, f"rollout-iter{it}"),
                           commit=runs.commit, task_transform=episode_task_for(it))
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
        rew = reward_summary(examples)
        live = sum(1 for ex in examples if ex.advantage != 0)
        # print liveness the MOMENT the rollout ends — before the slow update_step — so a dead
        # round (live=0) is visible without waiting for the gradient step
        print(f"iter {it:3d} rollout  live={live}/{len(examples)}  |  "
              f"attack_policy_leak={d['attack_policy_leak']:.0%}  "
              f"benign_policy_leak={d['benign_policy_leak']:.0%}  "
              f"protocol_fail={d['protocol_failure']:.0%}  "
              f"refuse={d['refuse']:.0%}  clean={d['clean']:.0%}  |  "
              f"dead_groups={div['dead_group_rate']:.0%} "
              f"mixed_reward={div['mixed_reward_group_rate']:.0%}  |  "
              # The scalar the policy is optimizing. Component rates trade against each other;
              # a flat mean with anti-correlated components means cycling, not learning.
              f"mean_reward={rew['mean_reward']:+.3f} "
              f"(atk={rew['mean_reward_attack']:+.3f} ben={rew['mean_reward_benign']:+.3f})",
              flush=True)

        model.train()
        m = update_step(learner, examples, beta=BETA, clip_eps=CLIP_EPS,
                        inner_epochs=INNER_EPOCHS)
        print(f"iter {it:3d} update   loss={m['loss']:.4f}  ratio={m['mean_ratio']:.3f}  "
              f"kl={m['mean_kl']:.5f}  grad={m['grad_norm']:.3f}  "
              f"epoch_ratios={[round(r, 3) for r in m['epoch_ratios']]}  "
              f"live_episodes={m['n_live_episodes']}  "
              f"live_decisions={m['n_live_examples']}/{m['n_examples']}", flush=True)

        # checkpoint EVERY iter so a preemption costs at most one iter of work. Save order is
        # load-bearing: adapter -> optimizer -> meta.json LAST (the completion marker). We commit
        # only AFTER the marker, so the Volume never exposes a half-written checkpoint to resume.
        # The next rollout must sample from the policy we just trained, not the previous one.
        publish_policy(f"iter{it} post-update")

        ckpt = os.path.join(CKPT_ROOT, f"iter{it}")
        model.save_pretrained(ckpt)
        torch.save(opt.state_dict(), os.path.join(ckpt, "optimizer.pt"))
        write_meta(ckpt, {"iter": it, "phase_identity": phase_identity})
        prune_checkpoints(CKPT_ROOT, keep=CKPT_KEEP)
        # The rollout bank is only disposable once THIS iteration's checkpoint exists: dropping it
        # earlier would make a preemption between update and checkpoint re-run the whole rollout.
        shutil.rmtree(os.path.join(CKPT_ROOT, f"rollout-iter{it}"), ignore_errors=True)
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


@app.function(gpu="A100-80GB", timeout=24 * 60 * 60,
              retries=modal.Retries(initial_delay=0.0, max_retries=10),
              volumes={"/cache/huggingface": hf_cache, "/runs": runs})
def train():
    """Own the vLLM lifecycle OUTSIDE the training body.

    Modal can retry a failed input in the SAME container. Without this, the previous attempt's
    servers stay alive holding ~52 GiB and the retry OOMs before it can even take a CUDA context
    -- which is exactly what happened. Every return and every exception must release them.
    """
    active_servers = []
    try:
        return _train_impl(active_servers)
    finally:
        for stop_server, proc in reversed(active_servers):
            stop_server(proc)


@app.local_entrypoint()
def main():
    train.remote()
