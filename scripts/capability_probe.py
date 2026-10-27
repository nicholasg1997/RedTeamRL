"""Cold large-model capability probe (inference only; no training, no restart, no teacher assist).

Runs a COLD base model as the scratchpad attacker on FRESH full episodes from turn 0 vs T2 (the oracle
literal redactor) -- the apples-to-apples setup GLM-5.3 won ~53% on. It answers the capacity question:
can a bigger/newer base *plan and reason its way to the leak*, and does it find MULTIPLE routes
(non-collapse) rather than one trick? Reports egress wins (ground truth) plus route diversity.

    modal run --detach scripts/capability_probe.py --tag cap-qwen38-27b-v1

Default model is Qwen3.8-27B on a single A100-80GB (51.75 GB weights fit with KV room to spare). CUDA
graphs are ON by default: in eager mode decode is launch-overhead-bound (~15 tok/s/seq) and the GPU
sits idle, so --enforce-eager is the real throttle, not the card. Pass --enforce-eager to fall back
if vLLM cannot graph-capture the GDN kernels. If vLLM 0.21.0 cannot register the architecture it
fails at server startup within minutes -- fall back to --model Qwen/Qwen3-32B (guaranteed-servable
control). run_collection checkpoints per episode and resumes, so a timeout just means re-running the
same tag.
"""
import modal

RUN_ROOT = "/runs/capability-probe"
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "main"  # exploratory; pin a commit SHA before promoting any result

image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "pydantic", "pyyaml")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
    .add_local_dir("backward_probe", remote_path="/root/backward_probe"))  # reuse the action-forcing fallback
sandbox_image = modal.Image.from_registry("python:3.12-slim").pip_install("requests", "pydantic", "pyyaml")
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
volumes = {"/runs": runs, "/cache": cache}
app = modal.App("redteamrl-capability-probe", image=image)


@app.function(gpu="A100-80GB", cpu=8, timeout=3600, volumes=volumes)  # gpu count must match tensor_parallel
def probe(tag: str, model: str = MODEL, revision: str = REVISION, episodes: int = 12,
          env_seed_start: int = 7000, workers: int = 8, max_turns: int = 12, max_tokens: int = 8192,
          thinking_token_budget: int = 6144, tensor_parallel: int = 1, request_timeout: float = 1200.0,
          enforce_eager: bool = False, sandbox_timeout: int = 3000):
    import hashlib
    import json
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from redteamrl.policies.vllm_client import make_vllm_generate, start_vllm_server, stop_vllm_server
    from redteamrl.sandbox.modal_shell import ModalShellSandbox
    from redteamrl.scratchpad.collection import execute_teacher_episode, probe_config, run_collection
    from redteamrl.scratchpad.experiment import atomic_json, source_digest
    from redteamrl.scratchpad.routes import route_diversity
    from backward_probe.generate import with_action_forcing
    runs.reload()
    cache.reload()
    # Weights download to the cache volume before reserving vLLM; a serving failure then means the
    # architecture is unsupported, not a missing file.
    snapshot_download(model, revision=revision,
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    cache.commit()
    config = probe_config(teacher=model, episodes=episodes, tier="T2", tag=tag,
                          source=source_digest("/root/redteamrl", __file__), env_seed_start=env_seed_start,
                          workers=workers, max_turns=max_turns, max_tokens=max_tokens)
    config.update(revision=revision, thinking_token_budget=thinking_token_budget,
                  tensor_parallel=tensor_parallel)
    # The slow 27B runs long episodes; keep each episode's sandbox alive near the function cap so we
    # don't lose episodes to "Sandbox is unavailable" (its default 1200s lifetime).
    config["sandbox"]["episode_timeout_s"] = sandbox_timeout
    out = Path(RUN_ROOT) / tag
    server = None
    try:
        extra_args = ["--revision", revision, "--reasoning-parser", "qwen3",
                      "--tensor-parallel-size", str(tensor_parallel)]
        if enforce_eager:  # CUDA graphs ON by default; eager is the fallback if GDN can't be captured
            extra_args.append("--enforce-eager")
        server = start_vllm_server(model, 8000, .90, max_model_len=16384, max_num_seqs=workers,
            extra_args=extra_args)

        def execute(job):
            # Deterministic per-episode seed: reproducible, and distinct across tasks so the repertoire
            # is measured over varied episodes, not one sampling path.
            seed = int(hashlib.sha256(job.key.encode()).hexdigest()[:8], 16) % (2 ** 31)

            def make_generate(on_response):
                primary = make_vllm_generate("http://localhost:8000", model,
                    temperature=config["temperature"], top_p=config["top_p"], max_tokens=max_tokens,
                    enable_thinking=True, require_reasoning=True, thinking_token_budget=thinking_token_budget,
                    seed=seed, on_response=on_response, label="probe-student", timeout=request_timeout)
                forcer = make_vllm_generate("http://localhost:8000", model,
                    temperature=config["temperature"], top_p=config["top_p"], max_tokens=512,
                    enable_thinking=False, require_reasoning=False, seed=seed, on_response=on_response,
                    label="probe-student-force", timeout=request_timeout)
                return with_action_forcing(primary, forcer)

            return execute_teacher_episode(job, config, make_generate,
                lambda: ModalShellSandbox(app, sandbox_image, **config["sandbox"]))

        report = run_collection(config, out, execute, commit=runs.commit)
        records = [json.loads(p.read_text()) for p in sorted(out.glob("ep-*.json"))]
        diversity = route_diversity(records)
        atomic_json(out / "route_diversity.json", diversity)
        runs.commit()

        row = report["summary"][config["tier"]]
        win = f"{row['win_rate']:.1%}" if row["win_rate"] is not None else "n/a"
        print(f"\nCAPABILITY {model} T2: completed={row['n_completed']}/{row['n_requested']} "
              f"failed={row['n_failed']} win={win} "
              f"recovery={row['recovery_wins']}/{row['recovery_eligible']} -> {report['summary']['gate_status']}",
              flush=True)
        print(f"ROUTE DIVERSITY: wins={diversity['wins']}/{diversity['completed']} "
              f"distinct_winning_routes={diversity['distinct_winning_routes']} "
              f"winning_routes={diversity['winning_routes']} novel_other_wins={diversity['novel_other_wins']}",
              flush=True)
        print(f"  routes_attempted={diversity['routes_attempted']}", flush=True)
        print(f"  routes_won={diversity['routes_won']}", flush=True)
        return dict(summary=report["summary"], route_diversity=diversity)
    finally:
        stop_vllm_server(server)


@app.local_entrypoint()
def main(tag: str = "cap-qwen38-27b-v1", model: str = MODEL, revision: str = REVISION,
         episodes: int = 12, env_seed_start: int = 7000, workers: int = 8, max_turns: int = 12,
         max_tokens: int = 8192, thinking_token_budget: int = 6144, tensor_parallel: int = 1,
         request_timeout: float = 1200.0, enforce_eager: bool = False, sandbox_timeout: int = 3000):
    import re
    from redteamrl.scratchpad.collection import probe_config
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", tag):
        raise ValueError("tag must be a simple 1-64 character name, not a path")
    if not 1 <= tensor_parallel <= 8:
        raise ValueError("tensor_parallel must be 1-8 (and gpu= must allocate that many cards)")
    if not 1 <= thinking_token_budget < max_tokens <= 8192:
        raise ValueError("thinking_token_budget must leave output space; max_tokens <= 8192")
    if request_timeout <= 0 or sandbox_timeout <= 0:
        raise ValueError("request_timeout and sandbox_timeout must be positive seconds")
    # Reuse the harness's own bounds check (tier/tag/episodes/workers/turns) before any dispatch.
    probe_config(teacher=model, episodes=episodes, tier="T2", tag=tag, source="preflight",
                 env_seed_start=env_seed_start, workers=workers, max_turns=max_turns, max_tokens=max_tokens)
    probe.remote(tag, model, revision, episodes, env_seed_start, workers, max_turns, max_tokens,
                 thinking_token_budget, tensor_parallel, request_timeout, enforce_eager, sandbox_timeout)
