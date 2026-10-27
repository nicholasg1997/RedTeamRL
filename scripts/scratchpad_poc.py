"""Small inference-only smoke test. No training and no automatic scale-up.

    .venv/bin/modal run scripts/scratchpad_poc.py --tag smoke-v1

Default: 6 attacks (2 worlds x 3 tiers), 24 benign checks, <=12 turns, full scratchpad.
Only after reviewing the smoke: --mode both --n-episodes 24 --benign-envs 8 --compiled
(`n-episodes` is PER MODE, so that comparison requests 48 attacks.)
"""
import modal

ATTACKER_MODEL = "Qwen/Qwen3-8B"
DEFENDER_MODEL = "Qwen/Qwen3-4B"
DEF_PORT, ATK_PORT = 8000, 8001
DEF_MEM_FRAC, ATK_MEM_FRAC = 0.25, 0.55  # no trainer; keep 20% GPU headroom
MAX_MODEL_LEN = 8192
DEF_MAX_TOKENS = 512
SCRATCHPAD_MAX_TOKENS = 768
ACTION_MAX_TOKENS = 1024
DEFENDER_CONTEXT_CHARS = 16_000
MAX_TURNS, WORKERS = 12, 8
ENV_SEED_START, CANARY_SEED, CANARY_REVISION = 5000, 0, 201
RUN_ROOT = "/runs/scratchpad-poc"

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "transformers<5", "pydantic", "pyyaml", "tqdm")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
          "VLLM_CACHE_ROOT": "/cache/inference/vllm",
          "TRITON_CACHE_DIR": "/cache/inference/triton",
          "CUDA_CACHE_PATH": "/cache/inference/cuda"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl",
                   ignore=["**/__pycache__/**", "**/.DS_Store"])
    .add_local_file("scripts/scratchpad_poc.py", "/root/scratchpad_poc_source.py")
)
prepare_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub<1.0", "pydantic")
    .env({"HF_HOME": "/cache/huggingface"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl",
                   ignore=["**/__pycache__/**", "**/.DS_Store"])
    .add_local_file("scripts/scratchpad_poc.py", "/root/scratchpad_poc_source.py")
)
shell_image = modal.Image.debian_slim(python_version="3.12").workdir("/workspace")
hf_cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
inference_cache = modal.Volume.from_name("redteamrl-inference-cache", create_if_missing=True)
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
app = modal.App("redteamrl-scratchpad-poc", image=image)


def make_config(n_episodes=6, max_turns=MAX_TURNS, mode="full", workers=WORKERS,
                benign_envs=2, compiled=False, planner_thinking=False, planner_max_tokens=0,
                planner_thinking_budget=1536, action_thinking=False, action_thinking_budget=256,
                single_call=True):
    if planner_thinking and mode == "tried-only":
        raise ValueError("planner thinking requires a scratchpad arm (full or both)")
    if not 1 <= planner_thinking_budget <= 3072:
        raise ValueError("planner-thinking-budget must be 1..3072")
    if not 1 <= action_thinking_budget <= 1024:
        raise ValueError("action-thinking-budget must be 1..1024")
    budget = planner_max_tokens or (planner_thinking_budget + SCRATCHPAD_MAX_TOKENS
                                   if planner_thinking else SCRATCHPAD_MAX_TOKENS)
    if not 256 <= budget <= 4096:
        raise ValueError("planner-max-tokens must be 256..4096 (or 0 for automatic)")
    if planner_thinking and budget - planner_thinking_budget < 256:
        raise ValueError("planner-max-tokens must leave at least 256 tokens after planner-thinking-budget")
    return dict(single_call=single_call, planner_thinking=planner_thinking,
                planner_response_protocol="budgeted-reasoning-v2",
                planner_thinking_budget=planner_thinking_budget if planner_thinking else None,
                action_thinking=action_thinking,
                action_thinking_budget=action_thinking_budget if action_thinking else None,
                n_episodes=n_episodes, max_turns=max_turns, mode=mode, workers=workers,
                benign_envs=benign_envs, compiled=compiled, attacker=ATTACKER_MODEL,
                defender=DEFENDER_MODEL, max_model_len=MAX_MODEL_LEN,
                scratchpad_max_tokens=budget,
                action_max_tokens=ACTION_MAX_TOKENS + (action_thinking_budget if action_thinking else 0),
                defender_max_tokens=DEF_MAX_TOKENS, defender_context_chars=DEFENDER_CONTEXT_CHARS,
                env_seed_start=ENV_SEED_START, canary_seed=CANARY_SEED,
                canary_revision=CANARY_REVISION, sandbox="isolated-modal",
                action_protocol="strict-json-v1", vllm="0.21.0", gpu="A100-80GB", memory_fractions=[DEF_MEM_FRAC, ATK_MEM_FRAC])


@app.function(image=prepare_image, cpu=2.0, timeout=20 * 60,
              volumes={"/cache/huggingface": hf_cache, "/runs": runs})
def prepare(config: dict, tag: str, retry_failed: bool = False):
    """Download/cache weights on CPU, pin revisions, reject incompatible resumes BEFORE GPU."""
    import json
    import sys
    from pathlib import Path
    from huggingface_hub import snapshot_download
    sys.path.insert(0, "/root")
    from redteamrl.scratchpad.experiment import ensure_manifest, source_digest, plan_jobs
    from redteamrl.scratchpad.gate import run_benign_gate
    from redteamrl.scratchpad.tiers import make_tiers
    from redteamrl.sandbox.modal_shell import ModalShellSandbox

    hf_cache.reload()
    runs.reload()
    config = dict(config, source_sha256=source_digest("/root/redteamrl", "/root/scratchpad_poc_source.py"))
    manifest_path = Path(RUN_ROOT) / tag / "manifest.json"
    previous = json.loads(manifest_path.read_text())["config"] if manifest_path.exists() else None
    if previous:
        # A resumed run uses the original model revisions even if the HF main branch has changed.
        comparable = {k: v for k, v in previous.items() if k != "model_revisions"}
        if comparable != config:
            raise ValueError("Run configuration/source changed: use a fresh --tag")
        summary_path = manifest_path.with_name("summary.json")
        if summary_path.exists() and not retry_failed:
            return {"config": previous, "summary": json.loads(summary_path.read_text())}

    # Exercise the real sandbox API and build its tiny image on CPU before reserving an A100.
    gate_jobs, _ = plan_jobs(config["n_episodes"], config["mode"], config["benign_envs"],
                            config["env_seed_start"], config["canary_seed"], config["canary_revision"])
    oracle = next(t for t in make_tiers() if t.name == "T2")
    probe_spec = next(j.spec for j in gate_jobs if j.spec.id.endswith("-mixed-read"))
    [probe] = run_benign_gate(oracle, [probe_spec], None,
                            lambda: ModalShellSandbox(app, shell_image, episode_timeout_s=120))
    if not probe.complete or probe.leak:
        raise RuntimeError("Sandbox preflight failed: oracle did not safely complete the benign task")

    def download(model):
        revision = previous["model_revisions"][model] if previous else "main"
        print(f"Preparing model cache: {model} (revision={revision})", flush=True)
        path = snapshot_download(model, revision=revision,
                                 allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
        # Persist each finished model even if the next download fails.
        hf_cache.commit()
        print(f"Model cache ready: {model}", flush=True)
        return model, Path(path).name

    # snapshot_download already downloads files concurrently. Concurrent outer calls
    # race on tqdm's class-level progress lock during thread_map cleanup.
    config["model_revisions"] = dict(download(model) for model in [ATTACKER_MODEL, DEFENDER_MODEL])
    ensure_manifest(f"{RUN_ROOT}/{tag}", config)
    runs.commit()
    return {"config": config, "summary": None}


@app.function(gpu="A100-80GB", cpu=8.0, timeout=30 * 60, retries=0, max_containers=1,
              volumes={"/cache/huggingface": hf_cache, "/cache/inference": inference_cache, "/runs": runs})
def run(config: dict, tag: str, retry_failed: bool = False):
    """30-minute execution limit per attempt; Modal may restart a preempted container.

    Explicit reruns resume persisted episode records. Application errors are not auto-retried.
    """
    import json
    import os
    import sys
    import time
    import traceback
    from pathlib import Path
    import requests
    sys.path.insert(0, "/root")
    from redteamrl.policies.vllm_client import make_vllm_generate, start_vllm_server, stop_vllm_server
    from redteamrl.sandbox.modal_shell import ModalShellSandbox
    from redteamrl.scratchpad.experiment import (
        atomic_json, ensure_manifest, gate_summary, generation_summary, plan_jobs, run_jobs, summarize,
    )
    from redteamrl.scratchpad.gate import run_benign_gate
    from redteamrl.scratchpad.poc import SCRATCHPAD_SYSTEM, SEED_NEXT_IDEAS, make_attacker_factory
    from redteamrl.scratchpad.rollout import run_scratchpad_rollout, to_jsonable
    from redteamrl.scratchpad.tiers import make_tiers

    hf_cache.reload()
    runs.reload()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    out_dir = f"{RUN_ROOT}/{tag}"
    ensure_manifest(out_dir, config)
    if Path(out_dir, "summary.json").exists() and not retry_failed:
        summary = json.loads(Path(out_dir, "summary.json").read_text())
        print("Saved run is complete; returning its report without starting model servers.", flush=True)
        return summary
    if retry_failed:
        # An interrupted explicit retry must not leave an old report masquerading as its result.
        Path(out_dir, "summary.json").unlink(missing_ok=True)
        runs.commit()

    gate_jobs, attack_jobs = plan_jobs(config["n_episodes"], config["mode"], config["benign_envs"],
                                      config["env_seed_start"], config["canary_seed"], config["canary_revision"])
    tiers = {t.name: t for t in make_tiers(max_context_chars=config["defender_context_chars"])}
    servers, timings = [], {}
    started = time.monotonic()

    def start(model, port, memory):
        before = time.monotonic()
        args = ["--revision", config["model_revisions"][model], "--tokenizer-revision",
                config["model_revisions"][model], "--enable-prefix-caching", "--enable-chunked-prefill"]
        if model == ATTACKER_MODEL and (config["planner_thinking"] or config["action_thinking"]):
            args += ["--reasoning-parser", "qwen3", "--reasoning-config",
                     '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}']
        if not config["compiled"]:
            args.append("--enforce-eager")
        # Sequential startup avoids overlapping vLLM memory profiling on the shared A100.
        servers.append(start_vllm_server(model, port, memory, max_model_len=config["max_model_len"],
                                        max_num_seqs=config["workers"], timeout_s=600, extra_args=args))
        timings[f"startup_{model}"] = time.monotonic() - before

    def execute(job):
        generations, agent, record = [], None, None
        sandbox = ModalShellSandbox(app, shell_image, episode_timeout_s=20 * 60)
        session = requests.Session()  # one per episode; no mutable transport shared across threads
        try:
            from redteamrl.harness.action_schema import action_schema

            def generate(label, model, port, tokens, temperature, section_check=True):
                thinking = ((label == "scratchpad" and config["planner_thinking"])
                            or (label == "action" and config["action_thinking"]))
                thinking_budget = (config["planner_thinking_budget"] if label == "scratchpad"
                                   else config["action_thinking_budget"] if label == "action" else None)
                base_generate = make_vllm_generate(f"http://localhost:{port}", model, max_tokens=tokens,
                                          temperature=temperature, seed=job.generation_seed,
                                          timeout=120.0, label=label, on_response=generations.append,
                                          session=session, enable_thinking=thinking, require_reasoning=thinking,
                                          thinking_token_budget=thinking_budget,
                                          json_schema=action_schema(job.spec.enabled_tools) if label == "action" else None)
                if not thinking or label != "scratchpad" or not section_check:
                    return base_generate

                def planner_generate(system, messages):
                    text = base_generate(system, messages)
                    from redteamrl.scratchpad.document import parse_scratchpad, TRIED_HEADING, GUIDANCE_HEADING
                    sections = parse_scratchpad(text).sections
                    if not any(body for name, body in sections.items()
                               if name not in {TRIED_HEADING, GUIDANCE_HEADING}):
                        raise RuntimeError("Planner returned no usable scratchpad sections; inspect saved generation")
                    return text
                return planner_generate


            defender_gen = generate("defender", DEFENDER_MODEL, DEF_PORT, config["defender_max_tokens"], 0.0)
            if job.kind == "benign":
                [result] = run_benign_gate(tiers[job.tier], [job.spec], defender_gen, lambda: sandbox)
                record = {"status": "completed", "result": result.model_dump()}
            else:
                if config["single_call"]:
                    def _unused(system, messages):
                        raise AssertionError("single-call mode must not invoke the action generate")
                    agent = make_attacker_factory(
                        generate("scratchpad", ATTACKER_MODEL, ATK_PORT,
                                 config["scratchpad_max_tokens"], 0.7, section_check=False),
                        _unused,
                        system_template=SCRATCHPAD_SYSTEM, seed_next_ideas=SEED_NEXT_IDEAS,
                        turn_budget=config["max_turns"], single_call=True,
                    )(job.spec)
                else:
                    agent = make_attacker_factory(
                        generate("scratchpad", ATTACKER_MODEL, ATK_PORT, config["scratchpad_max_tokens"], 0.7),
                        generate("action", ATTACKER_MODEL, ATK_PORT, config["action_max_tokens"], 0.7),
                        system_template=SCRATCHPAD_SYSTEM, seed_next_ideas=SEED_NEXT_IDEAS,
                        turn_budget=config["max_turns"], rewrite_scratchpad=job.mode == "full",
                        constrained_actions=True,
                    )(job.spec)
                rollout = run_scratchpad_rollout(job.spec, agent, tiers[job.tier].build(job.spec, defender_gen),
                                                 sandbox, max_turns=config["max_turns"], tier=job.tier)
                record = {"status": "completed", **to_jsonable(rollout)}
        except Exception as exc:
            record = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                      "traceback": traceback.format_exc()}
            if agent:
                from redteamrl.scratchpad.rollout import turn_to_jsonable
                record["partial_turns"] = [turn_to_jsonable(t) for t in agent.turn_records]
        finally:
            try:
                sandbox.close()
            except Exception as exc:
                record = {**(record or {}), "status": "failed", "cleanup_error": repr(exc)}
            session.close()
        return {**record, "generations": generations}

    try:
        start(DEFENDER_MODEL, DEF_PORT, DEF_MEM_FRAC)
        before = time.monotonic()
        benign = run_jobs(gate_jobs, execute, out_dir, workers=config["workers"],
                          commit=runs.commit, retry_failed=retry_failed)
        gates = gate_summary(benign)
        timings["benign_gate_s"] = time.monotonic() - before
        atomic_json(Path(out_dir) / "gates.json", gates)
        runs.commit()
        for tier, row in gates.items():
            print(f"{tier} gate={'PASS (provisional)' if row['valid'] else 'FAIL'} "
                  f"benign={row['n']}/{row['n_requested']} completion={row['benign_completion']:.1%} "
                  f"safe={row['safe_completion']} leak={row['benign_leak_rate']} "
                  f"errors={row['n_failed']} protocol={row['protocol_failure_episodes']}", flush=True)

        if any(gates[job.tier]["valid"] for job in attack_jobs):
            start(ATTACKER_MODEL, ATK_PORT, ATK_MEM_FRAC)
        before = time.monotonic()

        def gated_execute(job):
            if not gates[job.tier]["valid"]:
                return {"status": "skipped", "reason": "benign usability gate failed"}
            return execute(job)

        episodes = run_jobs(attack_jobs, gated_execute, out_dir, workers=config["workers"],
                            commit=runs.commit, retry_failed=retry_failed)
        timings["attack_phase_s"] = time.monotonic() - before
        summary = summarize(gates, episodes)
        summary["benign_generations"] = generation_summary(benign)
        summary["config"] = config
        summary["timings_this_invocation_s"] = timings
        timings["through_report_s"] = time.monotonic() - started
        atomic_json(Path(out_dir) / "summary.json", summary)
        runs.commit()
        for arm, row in summary["arms"].items():
            win = "N/A" if row["win_rate"] is None else f"{row['win_rate']:.1%}"
            print(f"{arm} completed={row['n_completed']}/{row['n_requested']} "
                  f"failed={row['n_failed']} skipped={row['n_skipped']} win={win} "
                  f"recovered={row['n_recovered']}/{row['n_redacted']} "
                  f"invalid={row['invalid_action_rate']:.1%} validation_errors={row['n_validation_errors']} looping={row['looping']:.1%} "
                  f"operational_ok={row['operational_ok']}", flush=True)
            for label, gen in row["generations"].items():
                print(f"  {label}: calls={gen['calls']} output_tokens={gen['completion_tokens']} "
                      f"length_finishes={gen['length_finishes']} max_prompt={gen['max_prompt_tokens']} "
                      f"thinking_calls={gen['thinking_calls']} missing_reasoning={gen['missing_reasoning']} "
                      f"missing_final={gen['missing_final']}", flush=True)
        return summary
    except Exception as exc:
        atomic_json(Path(out_dir) / "run_error.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        runs.commit()
        raise
    finally:
        for proc in reversed(servers):
            try:
                stop_vllm_server(proc)
            except Exception as exc:
                print(f"Model-server shutdown failed: {exc!r}", flush=True)
        try:
            inference_cache.commit()
        except Exception as exc:
            print(f"Inference-cache commit failed (episode results already persisted): {exc!r}", flush=True)
        print(f"GPU handler elapsed: {time.monotonic() - started:.1f}s", flush=True)


@app.local_entrypoint()
def main(n_episodes: int = 6, max_turns: int = MAX_TURNS, tag: str = "smoke-v1",
         mode: str = "full", workers: int = WORKERS, benign_envs: int = 2,
         compiled: bool = False, retry_failed: bool = False,
         planner_thinking: bool = False, planner_max_tokens: int = 0,
         planner_thinking_budget: int = 1536, action_thinking: bool = False,
         action_thinking_budget: int = 256, single_call: bool = True):
    import json
    from redteamrl.scratchpad.experiment import plan_jobs, validate_options
    validate_options(n_episodes, max_turns, tag, mode, workers, benign_envs)
    config = make_config(n_episodes, max_turns, mode, workers, benign_envs, compiled,
                         planner_thinking, planner_max_tokens, planner_thinking_budget,
                         action_thinking, action_thinking_budget, single_call)
    gates, attacks = plan_jobs(n_episodes, mode, benign_envs)
    print(f"Planned: {len(gates)} benign + up to {len(attacks)} attack episodes; "
          f"{max_turns} turns; mode={mode}; compiled={compiled}; planner_thinking={planner_thinking}; "
          f"planner_total_token_cap={config['scratchpad_max_tokens']}; "
          f"planner_thinking_budget={config['planner_thinking_budget']}; "
          f"action_thinking={action_thinking}; action_thinking_budget={config['action_thinking_budget']}; "
          f"action_total_token_cap={config['action_max_tokens']}. "
          "One A100, 30-minute execution limit per attempt, no application-error retries or training.", flush=True)
    prepared = prepare.remote(config, tag, retry_failed)
    summary = prepared["summary"]
    if summary is None:
        summary = run.remote(prepared["config"], tag, retry_failed)
    else:
        print("Returning saved report; no GPU allocated.")
    print(json.dumps(summary, indent=2))
    print(f"Report: redteamrl-eval-runs / scratchpad-poc/{tag}/summary.json")
