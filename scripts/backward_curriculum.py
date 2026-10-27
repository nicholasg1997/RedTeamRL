"""Phase 1 only: CPU-verified restarts, then cold Qwen success-by-depth. No training/GLM calls."""
import modal

ROOT = "/runs/backward-curriculum"
MODEL = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "pydantic", "pyyaml")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
    .add_local_dir("teacher_sft", remote_path="/root/teacher_sft")
    .add_local_dir("backward_probe", remote_path="/root/backward_probe"))
sandbox_image = modal.Image.from_registry("python:3.12-slim").pip_install("requests", "pydantic", "pyyaml")
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
volumes = {"/runs": runs, "/cache": cache}
app = modal.App("redteamrl-backward-curriculum", image=image)


def source_identity():
    from redteamrl.scratchpad.experiment import source_digest
    from teacher_sft.data import digest
    return digest([source_digest(f"/root/{p}", __file__)
                   for p in ("redteamrl", "teacher_sft", "backward_probe")])


def sandbox_factory(options):
    from backward_probe.sandbox import RestartSandbox
    return RestartSandbox(app, sandbox_image, **options)


@app.function(cpu=4, timeout=3600, volumes=volumes)
def prepare(tag: str, harvest_tag: str = "glm53-harvest", demos: int = 4,
            max_depth: int = 3, samples: int = 4):
    from huggingface_hub import snapshot_download
    from backward_probe.data import prepare
    runs.reload()
    cache.reload()
    report = prepare(f"/runs/gate0/{harvest_tag}", f"{ROOT}/{tag}",
        dict(source=source_identity(), harvest_tag=harvest_tag, demos=demos, max_depth=max_depth,
             samples=samples, model=MODEL, revision=REVISION), sandbox_factory, commit=runs.commit)
    if not report["ready"]:
        raise ValueError("No restart passed replay; see preparation.json. No GPU/model download launched.")
    # Weights transfer on CPU after successful replay, never while reserving the A100.
    snapshot_download(MODEL, revision=REVISION,
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    cache.commit()
    print(report, flush=True)
    return report


@app.function(cpu=1, timeout=60, volumes={"/runs": runs})
def check_ready(tag: str):
    from backward_probe.data import load_prepared
    runs.reload()
    manifest, _, report = load_prepared(f"{ROOT}/{tag}")
    if manifest["config"]["source"] != source_identity():
        raise ValueError("Code changed since preparation; use a new tag")
    return report


@app.function(gpu="A100-80GB", cpu=8, timeout=3600, volumes=volumes)
def diagnose(tag: str, workers: int = 4, max_tokens: int = 4096, thinking_token_budget: int = 3072,
             unassisted_only: bool = False):
    from transformers import AutoTokenizer
    from redteamrl.policies.vllm_client import make_vllm_generate, start_vllm_server, stop_vllm_server
    from backward_probe.data import load_prepared
    from backward_probe.runner import ARMS, generation_config, run
    runs.reload()
    cache.reload()
    root = f"{ROOT}/{tag}"
    manifest, _, _ = load_prepared(root)
    if manifest["config"]["source"] != source_identity():
        raise ValueError("Code changed since preparation; use a new tag")
    config = generation_config(workers=workers, max_tokens=max_tokens,
        thinking_token_budget=thinking_token_budget, model=MODEL, revision=REVISION,
        source=source_identity(), adapter=None, arms=("unassisted",) if unassisted_only else ARMS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    server = None
    try:
        server = start_vllm_server(MODEL, 8000, .85, max_model_len=config["max_model_len"],
            max_num_seqs=workers, extra_args=["--revision", REVISION,
                                             "--reasoning-parser", "qwen3", "--enforce-eager"])
        def generate(seed, callback):
            from backward_probe.generate import with_action_forcing
            primary = make_vllm_generate("http://localhost:8000", MODEL, temperature=config["temperature"],
                top_p=config["top_p"], max_tokens=max_tokens, enable_thinking=True, require_reasoning=True,
                thinking_token_budget=thinking_token_budget, seed=seed, on_response=callback,
                label="backward-student")
            # Action-forcing fallback: a budget-overrun thinking turn forces one thinking-OFF action
            # instead of discarding the continuation (dropping it biases the deep-restart read optimistic).
            forcer = make_vllm_generate("http://localhost:8000", MODEL, temperature=config["temperature"],
                top_p=config["top_p"], max_tokens=512, enable_thinking=False, require_reasoning=False,
                seed=seed, on_response=callback, label="backward-student-force")
            return with_action_forcing(primary, forcer)
        report = run(root, config, generate, sandbox_factory, tokenizer, commit=runs.commit)
        for depth, row in report["by_depth"].items():
            for arm in config["arms"]:
                r = row[arm]
                print(f"depth={depth} {arm} wins={r['wins']}/{r['completed']} failed={r['failed']} "
                      f"replay_skipped={r['replay_skipped']} exact_submits={r.get('exact_submits')} "
                      f"mean_sim={r.get('mean_edit_similarity')} max_sim={r.get('max_edit_similarity')}", flush=True)
            fmt = row.get("by_format_unassisted", {})
            print(f"  depth={depth} unassisted by_format: " + (" ".join(
                f"{m}[{v['bucket']}] wins={v['wins']}/{v['n']} sim={v['mean_edit_similarity']}"
                for m, v in fmt.items()) or "(none)"), flush=True)
        return report
    finally:
        stop_vllm_server(server)


@app.local_entrypoint()
def main(stage: str = "prepare", tag: str = "backward-v1", harvest_tag: str = "glm53-harvest",
         demos: int = 4, max_depth: int = 3, samples: int = 4, workers: int = 4,
         max_tokens: int = 4096, thinking_token_budget: int = 3072, unassisted_only: bool = False):
    import re
    from backward_probe.runner import ARMS, generation_config
    for name in (tag, harvest_tag):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise ValueError("Tags must be simple names, not paths")
    if not 1 <= demos <= 16 or not 1 <= max_depth <= 12 or not 1 <= samples <= 8:
        raise ValueError("Bounds: demos 1-16, max_depth 1-12, samples 1-8")
    generation_config(workers=workers, max_tokens=max_tokens, thinking_token_budget=thinking_token_budget,
                      arms=("unassisted",) if unassisted_only else ARMS)
    if stage == "prepare":
        prepare.remote(tag, harvest_tag, demos, max_depth, samples)
    elif stage == "run":
        print(check_ready.remote(tag), flush=True)
        diagnose.remote(tag, workers, max_tokens, thinking_token_budget, unassisted_only)
    else:
        raise ValueError("stage must be prepare or run")
