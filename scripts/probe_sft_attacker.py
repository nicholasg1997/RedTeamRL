"""One-off diagnostic: WHY does the SFT'd attacker collapse to the accidental-leak floor (2.3%)?

Not part of training. Replays a few HELD-OUT attack episodes with the real frozen defender, once
with the BASE attacker and once with the SFT'd attacker (base + the saved LoRA at
CKPT_ROOT/adapter), capturing EVERY attacker generation's raw text, finish_reason, and token
counts -- the after-eval serving path exactly (same model, same 2048 max_tokens, same temp 0.7).

The leading hypothesis is that SFT made the reasoning verbose enough to blow the 2048-token budget
before the closing JSON action, so `parse_attacker_action` finds nothing and the turn is forfeited
(kind="message", text=_INVALID_ATTACKER_OUTPUT). That is a *deploy-realistic emission* failure, not
a competence loss, and it would floor the win rate regardless of the adapter's content -- which is
why two different SFT designs both landed on the identical fixed-seed floor.

This prints, per policy: truncation rate, completion-token distribution, valid-emission rate, action
kind mix, submit count, and episode win rate -- THEN dumps a handful of full raw generations so the
failure mode is legible, not just statistical.

Run: modal run scripts/probe_sft_attacker.py::probe
"""
import modal

# ---- mirror scripts/sft_attacker.py so the serving path is byte-for-byte the after-eval's ----
ATTACKER_MODEL = "Qwen/Qwen3-8B"
DEFENDER_MODEL = "Qwen/Qwen3-4B"
DEFENDER_SFT_ADAPTER = "/runs/sft-defender/round9/training/7eeae96976a4/adapter"
DEFENDER_GRPO_ADAPTER = "/runs/defender-grpo-r9-vllm/iter1"
CKPT_ROOT = "/runs/attacker-sft"
ADAPTER_DIR = f"{CKPT_ROOT}/adapter"

ATK_PORT, DEF_PORT = 8000, 8001
ATK_MEM_FRAC, DEF_MEM_FRAC = 0.45, 0.15
ATK_MAX_MODEL_LEN, DEF_MAX_MODEL_LEN = 16384, 8192
ATK_MAX_NUM_SEQS = DEF_MAX_NUM_SEQS = 16
ATK_MAX_TOKENS, DEF_MAX_NEW_TOKENS = 2048, 512
TEMP, TOP_P = 0.7, 0.8
TRANSCRIPT_CONTEXT_CHARS = 40_000
MAX_TURNS = 12
REDACTION_ENFORCEMENT = "unshielded"
CANARY_SEED, CANARY_REVISION = 0, 101

# Small on purpose -- this is a read-the-generations probe, not a metric. A handful of episodes per
# held-out task is enough to see the emission mode; raise if you want tighter stats.
N_EPISODES_PER_TASK = 2
N_DUMP = 6                     # full raw generations to print per policy

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "peft>=0.11",
                 "accelerate>=0.30", "pydantic", "pyyaml", "tqdm")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
)
hf_cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
app = modal.App("redteamrl-attacker-sft-probe", image=image)


def _merge_frozen_defender(tmp):
    """base -> SFT -> GRPO, merged and saved -- identical to sft_attacker's frozen opponent."""
    import tempfile
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    adapters = [a for a in (DEFENDER_SFT_ADAPTER, DEFENDER_GRPO_ADAPTER) if a]
    merged_dir = tempfile.mkdtemp(prefix="frozen-defender-", dir=tmp)
    model = AutoModelForCausalLM.from_pretrained(
        DEFENDER_MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
    for adapter in adapters:                                  # order matters: SFT then GRPO
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(DEFENDER_MODEL).save_pretrained(merged_dir)
    del model
    return merged_dir


def _capturing_generate(base_url, model, sink_holder):
    """A generate(system, messages) -> str that ALSO records finish_reason + token usage.

    make_vllm_generate returns only the message content, and finish_reason (did we hit the token
    cap?) is the whole diagnosis here, so we call /v1/chat/completions directly. `sink_holder` is a
    one-key dict whose "sink" is the current episode's per-turn list (set by the caller before each
    episode); sequential replay means no thread-safety dance.
    """
    import requests
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    def generate(system: str, messages: list) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": TEMP, "top_p": TOP_P, "max_tokens": ATK_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM {resp.status_code} from {url}: {resp.text[:800]}")
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage") or {}
        sink = sink_holder.get("sink")
        if sink is not None:
            sink.append({
                "raw": content,
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            })
        return content

    return generate


def _probe(active_servers):
    import os, sys, statistics, tempfile
    from collections import Counter
    sys.path.insert(0, "/root")
    from redteamrl.envs.registry import held_out_tasks
    from redteamrl.harness.episode import run_episode
    from redteamrl.harness.protocol import (
        parse_attacker_action, _extract_json, _INVALID_ATTACKER_OUTPUT)
    from redteamrl.policies.prompted import (
        PromptedAttacker, PromptedDefender, ATTACKER_SYSTEM_SHELL, HINT_TECHNIQUE)
    from redteamrl.policies.vllm_client import (
        make_vllm_generate, start_vllm_server, stop_vllm_server)
    from redteamrl.sandbox.shell import LocalShellSandbox
    from redteamrl.sft.canary import randomize_task_canary
    from redteamrl.attack_sft.eval_metrics import (
        defender_decision_breakdown, winning_action_sequences)

    if not os.path.isdir(ADAPTER_DIR):
        raise RuntimeError(f"no SFT adapter at {ADAPTER_DIR} -- run scripts/sft_attacker.py::train first")

    tmp = tempfile.mkdtemp(prefix="probe-", dir="/tmp")
    held_attack = [t for t in held_out_tasks() if t.episode_type == "attack"]
    print(f"held-out attack tasks: {[t.id for t in held_attack]}  "
          f"({N_EPISODES_PER_TASK} episodes each)", flush=True)

    # ---- frozen defender up first, identical to the after-eval ----
    merged_defender = _merge_frozen_defender(tmp)
    active_servers.append((stop_vllm_server, start_vllm_server(
        merged_defender, DEF_PORT, DEF_MEM_FRAC, max_model_len=DEF_MAX_MODEL_LEN,
        max_num_seqs=DEF_MAX_NUM_SEQS)))
    gen_def = make_vllm_generate(f"http://localhost:{DEF_PORT}", merged_defender,
                                 enable_thinking=False, max_tokens=DEF_MAX_NEW_TOKENS)

    def defender_factory():
        return PromptedDefender(generate=gen_def, max_context_chars=TRANSCRIPT_CONTEXT_CHARS)

    def replay(gen, sink_holder):
        """Run the held-out episodes sequentially; return per-episode captured turns + win flag."""
        episodes = []
        for spec in held_attack:
            for r in range(N_EPISODES_PER_TASK):
                episode_spec = randomize_task_canary(spec, r, CANARY_SEED, CANARY_REVISION)
                agent = PromptedAttacker(
                    gen, goal=episode_spec.goal, cover=episode_spec.request or "",
                    hint=HINT_TECHNIQUE, system_template=ATTACKER_SYSTEM_SHELL,
                    max_context_chars=TRANSCRIPT_CONTEXT_CHARS)
                turns = []
                sink_holder["sink"] = turns
                sandbox = LocalShellSandbox()
                try:
                    result = run_episode(episode_spec, agent, defender_factory(), sandbox,
                                         max_turns=MAX_TURNS,
                                         redaction_enforcement=REDACTION_ENFORCEMENT)
                finally:
                    sandbox.close()
                    sink_holder["sink"] = None
                episodes.append({"task_id": episode_spec.id, "won": bool(result.leak),
                                 "turns": turns, "steps": list(result.steps)})
        return episodes

    def analyze(episodes, label):
        turns = [t for ep in episodes for t in ep["turns"]]
        n = max(len(turns), 1)
        trunc = sum(1 for t in turns if t["finish_reason"] == "length")
        comps = [t["completion_tokens"] for t in turns if t["completion_tokens"] is not None]
        kinds, valid, submits = Counter(), 0, 0
        for t in turns:
            action = parse_attacker_action(t["raw"])
            failed = action.kind == "message" and action.text == _INVALID_ATTACKER_OUTPUT
            if failed:
                kinds["<forfeit: no valid action>"] += 1
            else:
                valid += 1
                kinds[action.kind] += 1
                if action.kind == "submit":
                    submits += 1
        wins = sum(1 for ep in episodes if ep["won"])
        pen = lambda k: f"{k}/{len(turns)} ({100*k/n:.0f}%)"

        def q(p):
            return int(statistics.quantiles(comps, n=10)[p]) if len(comps) > 1 else (comps[0] if comps else 0)

        print(f"\n========== {label}  ({len(episodes)} episodes) ==========", flush=True)
        print(f"  episode win rate (leak):    {wins}/{len(episodes)} ({100*wins/max(len(episodes),1):.1f}%)")
        print(f"  attacker generations:       {len(turns)}")
        print(f"  TRUNCATED (finish=length):  {pen(trunc)}   <-- emission-failure smoking gun")
        print(f"  valid emitted actions:      {pen(valid)}")
        print(f"  submits emitted:            {submits}")
        print(f"  action kinds:               {dict(kinds)}")
        if comps:
            print(f"  completion tokens:          mean={statistics.mean(comps):.0f}  "
                  f"median={statistics.median(comps):.0f}  p90={q(8)}  max={max(comps)}  "
                  f"(cap={ATK_MAX_TOKENS})")
        return turns

    def dump(turns, label, k):
        # Prefer to SHOW the failure: truncated turns first, then the rest, so the reader sees the
        # mode rather than a lucky good turn. Full raw text (bounded by max_tokens anyway).
        ordered = sorted(range(len(turns)),
                         key=lambda i: (turns[i]["finish_reason"] != "length", i))
        print(f"\n########## {label}: {min(k, len(turns))} full generations ##########", flush=True)
        for rank, i in enumerate(ordered[:k]):
            t = turns[i]
            action = parse_attacker_action(t["raw"])
            failed = action.kind == "message" and action.text == _INVALID_ATTACKER_OUTPUT
            has_json = _extract_json(t["raw"]) is not None
            print(f"\n----- {label} gen #{rank} | finish={t['finish_reason']} "
                  f"completion_tokens={t['completion_tokens']} | trailing_json={has_json} "
                  f"| parsed={'FORFEIT' if failed else action.kind} -----", flush=True)
            print(t["raw"], flush=True)

    # ---- BASE attacker (no adapter) ----
    print("\n>>> serving BASE attacker...", flush=True)
    base_holder = {"sink": None}
    base_proc = start_vllm_server(ATTACKER_MODEL, ATK_PORT, ATK_MEM_FRAC,
                                  max_model_len=ATK_MAX_MODEL_LEN, max_num_seqs=ATK_MAX_NUM_SEQS)
    active_servers.append((stop_vllm_server, base_proc))
    base_gen = _capturing_generate(f"http://localhost:{ATK_PORT}", ATTACKER_MODEL, base_holder)
    base_eps = replay(base_gen, base_holder)
    base_turns = analyze(base_eps, "BASE attacker")
    base_steps = [s for ep in base_eps for s in ep["steps"]]
    bd = defender_decision_breakdown(base_steps)
    n = max(bd["n_tool_calls"], 1)
    print("\n========== DEFENDER DECISION BREAKDOWN (base episodes) ==========", flush=True)
    print(f"  tool calls: {bd['n_tool_calls']}", flush=True)
    print(f"  rejected:        {bd['reject']}/{bd['n_tool_calls']} ({100*bd['reject']/n:.0f}%)", flush=True)
    print(f"  allow+redacted:  {bd['allow_redacted']}/{bd['n_tool_calls']} ({100*bd['allow_redacted']/n:.0f}%)", flush=True)
    print(f"  allow+CLEAN:     {bd['allow_clean']}/{bd['n_tool_calls']} ({100*bd['allow_clean']/n:.0f}%)"
          "   <-- validated-good actions Option B will yield", flush=True)
    print("\n########## WINNING STRATEGIES (base episodes) ##########", flush=True)
    seqs = winning_action_sequences(
        [{"won": ep["won"], "turns": ep["steps"]} for ep in base_eps])
    if not seqs:
        print("  (no base wins in this sample — raise N_EPISODES_PER_TASK)", flush=True)
    for i, seq in enumerate(seqs):
        print(f"  win #{i}: " + " -> ".join(str(k) for k in seq), flush=True)
    stop_vllm_server(base_proc)
    active_servers.remove((stop_vllm_server, base_proc))

    # ---- SFT'd attacker (base + saved LoRA) ----
    print("\n>>> serving SFT'd attacker (base + adapter)...", flush=True)
    sft_holder = {"sink": None}
    sft_proc = start_vllm_server(ATTACKER_MODEL, ATK_PORT, ATK_MEM_FRAC,
                                 max_model_len=ATK_MAX_MODEL_LEN, max_num_seqs=ATK_MAX_NUM_SEQS,
                                 lora_name="attacker-sft", lora_path=ADAPTER_DIR)
    active_servers.append((stop_vllm_server, sft_proc))
    sft_gen = _capturing_generate(f"http://localhost:{ATK_PORT}", "attacker-sft", sft_holder)
    sft_eps = replay(sft_gen, sft_holder)
    sft_turns = analyze(sft_eps, "SFT'd attacker")

    # ---- read-the-gens dumps (SFT first: that is what we are debugging) ----
    dump(sft_turns, "SFT'd attacker", N_DUMP)
    dump(base_turns, "BASE attacker", max(2, N_DUMP // 2))
    print("\n[probe done]", flush=True)


@app.function(gpu="A100-80GB", timeout=60 * 60,
              volumes={"/cache/huggingface": hf_cache, "/runs": runs})
def probe():
    from redteamrl.policies.vllm_client import stop_vllm_server  # noqa: F401 (imported for parity)
    active_servers = []
    try:
        _probe(active_servers)
    finally:
        # Preemptible: always release every vLLM process so a retry lands on a clean GPU.
        for stop, proc in active_servers:
            try:
                stop(proc)
            except Exception as exc:                           # noqa: BLE001
                print(f"[cleanup] {type(exc).__name__}: {exc}", flush=True)
