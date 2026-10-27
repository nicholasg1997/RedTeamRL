"""Bounded teacher warm-start stages. See teacher_sft/README.md for run order.

No stage launches another paid stage automatically. All functions retain the one-hour ceiling.
New source files live outside redteamrl so the in-flight harvest stays resumable.
"""
import modal

MODEL = "Qwen/Qwen3-14B"
ROOT = "/runs/teacher-sft"
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "peft>=0.11",
                 "accelerate>=0.30", "pydantic", "pyyaml", "tqdm")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
    .add_local_dir("teacher_sft", remote_path="/root/teacher_sft")
)
# Sandboxes only need the existing harness/runtime, not a CUDA or model-serving image.
sandbox_image = modal.Image.from_registry("python:3.12-slim").pip_install("pydantic", "pyyaml")
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
app = modal.App("redteamrl-teacher-sft", image=image)
volumes = {"/runs": runs, "/cache": cache}


def source_identity():
    from redteamrl.scratchpad.experiment import source_digest
    from teacher_sft.data import digest
    return digest([source_digest("/root/redteamrl", __file__),
                   source_digest("/root/teacher_sft", __file__)])


def serving_config(revision):
    return dict(model=MODEL, revision=revision, source=source_identity(), seed=42,
                temperature=0.6, top_p=0.95, max_tokens=8192, thinking_token_budget=6144,
                max_model_len=16384, workers=4, enable_thinking=True)


def start_server(config, adapter=None):
    from redteamrl.policies.vllm_client import start_vllm_server
    return start_vllm_server(config["model"], 8000, .85, max_model_len=config["max_model_len"],
        max_num_seqs=config["workers"], lora_name="student" if adapter else None,
        lora_path=adapter, max_lora_rank=16,
        extra_args=["--revision", config["revision"], "--reasoning-parser", "qwen3", "--enforce-eager"])


def generate_for(config, callback, *, seed=None, adapter=None):
    from redteamrl.policies.vllm_client import make_vllm_generate
    model = "student" if adapter else config["model"]
    s = config["seed"] if seed is None else seed
    primary = make_vllm_generate("http://localhost:8000", model,
        temperature=config["temperature"], top_p=config["top_p"], max_tokens=config["max_tokens"],
        enable_thinking=True, require_reasoning=True, thinking_token_budget=config["thinking_token_budget"],
        seed=s, label="student", on_response=callback)
    # Action-forcing fallback: if the student's reasoning overruns the budget and leaves no action,
    # force one with a thinking-OFF call (not a new plan -- just emit the act its analysis implied).
    # Mirrors the teacher-harvest repair; a long-thinking turn no longer fails the whole episode.
    forcer = make_vllm_generate("http://localhost:8000", model,
        temperature=config["temperature"], top_p=config["top_p"], max_tokens=512,
        enable_thinking=False, require_reasoning=False, seed=s, label="student-force", on_response=callback)
    def generate(system, messages):
        try:
            return primary(system, messages)
        except RuntimeError as exc:
            m = str(exc)
            if "reasoning" not in m or ("missing" not in m and "did not finish" not in m):
                raise
            forced = [dict(x) for x in messages]
            forced[-1] = {**forced[-1], "content": forced[-1]["content"]
                          + "\n\nYour reasoning was cut off. Output ONLY the single JSON action now, no thinking."}
            return forcer(system, forced)
    return generate


@app.function(cpu=2, timeout=3600, volumes=volumes)
def prepare(harvest_tag: str, tag: str):
    from teacher_sft.data import prepare
    runs.reload()
    result = prepare(f"/runs/gate0/{harvest_tag}", f"{ROOT}/{tag}/prepared")
    runs.commit()
    print(result, flush=True)
    return result


@app.function(cpu=2, timeout=3600, volumes=volumes)
def resolve_model(tag: str):
    """Pin and cache on CPU before reserving a GPU; no concurrent snapshot_download race."""
    import json
    from pathlib import Path
    from huggingface_hub import model_info, snapshot_download
    from redteamrl.scratchpad.experiment import atomic_json
    from teacher_sft.data import load_prepared
    runs.reload()
    load_prepared(f"{ROOT}/{tag}/prepared")
    path = Path(f"{ROOT}/{tag}/student-model.json")
    if path.exists():
        revision = json.loads(path.read_text())["revision"]
    else:
        revision = model_info(MODEL).sha
        atomic_json(path, {"model": MODEL, "revision": revision})
        runs.commit()
    snapshot_download(MODEL, revision=revision, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    cache.commit()
    return revision


@app.function(gpu="A100-80GB", cpu=8, timeout=3600, volumes=volumes)
def author(tag: str, revision: str, max_candidates: int = 128):
    from teacher_sft.author import author_dataset
    from redteamrl.policies.vllm_client import stop_vllm_server
    runs.reload()
    cache.reload()
    config = {**serving_config(revision), "max_candidates": max_candidates}
    server = start_server(config)
    try:
        def generate(system, prompt):
            events = []
            try:
                generate_for(config, events.append)(system, [{"role": "user", "content": prompt}])
            except Exception:
                if not events:
                    raise
                # Preserve an incomplete response's actual metadata/content for auditing.
                # author_dataset independently rejects it; it never becomes a target.
            event = events[-1]
            return event.get("reasoning"), event.get("final_content"), event
        return author_dataset(f"{ROOT}/{tag}/prepared", f"{ROOT}/{tag}/authored", generate,
                              config, commit=runs.commit)
    finally:
        stop_vllm_server(server)


@app.function(gpu="A100-80GB", cpu=8, timeout=3600, volumes=volumes)
def train(tag: str, arm: str = "wins", max_steps: int = 20, epochs: int = 1,
          max_target_tokens: int = 60000, rank: int = 8, learning_rate: float = 3e-5):
    import json
    from pathlib import Path
    from teacher_sft.train import train_adapter
    runs.reload()
    cache.reload()
    model = json.loads(Path(f"{ROOT}/{tag}/student-model.json").read_text())
    if not 1 <= epochs <= 10 or not 1 <= rank <= 128 or learning_rate <= 0 or max_target_tokens < 1:
        raise ValueError("train bounds: epochs 1-10, rank 1-128, lr>0, max_target_tokens>=1")
    return train_adapter(f"{ROOT}/{tag}/authored", f"{ROOT}/{tag}/train-{arm}",
                         {**model, "source": source_identity(), "arm": arm, "max_steps": max_steps,
                          "epochs": epochs, "max_target_tokens": max_target_tokens,
                          "rank": rank, "alpha": 2 * rank, "learning_rate": learning_rate},
                         commit=runs.commit)


@app.function(gpu="A100-80GB", cpu=8, timeout=3600, volumes=volumes)
def evaluate(tag: str, arm: str = "base", n_generated: int = 8, workers: int = 12):
    import json
    from pathlib import Path
    from teacher_sft.evaluate import evaluate, evaluation_jobs
    from teacher_sft.data import digest
    from teacher_sft.train import latest_checkpoint
    from redteamrl.policies.vllm_client import stop_vllm_server
    from redteamrl.sandbox.modal_shell import ModalShellSandbox
    runs.reload()
    cache.reload()
    model = json.loads(Path(f"{ROOT}/{tag}/student-model.json").read_text())
    config = {**serving_config(model["revision"]), "n_generated": n_generated, "max_turns": 12,
              "seed": 2027, "sandbox": dict(timeout_s=10, output_cap=4096, episode_timeout_s=1200)}
    if not 1 <= workers <= 16:
        raise ValueError("workers must be 1-16")
    config["workers"] = workers  # override serving_config default (4): raises vLLM max_num_seqs and the episode pool together
    adapter = None
    if arm != "base":
        training = Path(f"{ROOT}/{tag}/train-{arm}")
        report = json.loads((training / "report.json").read_text())  # Finished bounded training required.
        adapter = str(latest_checkpoint(training))
        meta = json.loads((Path(adapter) / "meta.json").read_text())
        if meta["model"] != MODEL or meta["revision"] != model["revision"] or meta["identity"] != report["identity"]:
            raise ValueError("Adapter/model identity mismatch")
        config["adapter_identity"] = digest(meta)
    config["adapter"] = adapter
    evaluation_jobs(f"{ROOT}/{tag}/prepared", n_generated, config["seed"])  # Fail before vLLM startup.
    server = start_server(config, adapter)
    try:
        return evaluate(f"{ROOT}/{tag}/prepared", f"{ROOT}/{tag}/eval-{arm}", config,
            lambda job, callback: generate_for(config, callback, seed=job.generation_seed, adapter=adapter),
            lambda: ModalShellSandbox(app, sandbox_image, **config["sandbox"]), commit=runs.commit)
    finally:
        stop_vllm_server(server)


@app.function(cpu=2, timeout=3600, volumes=volumes)
def compare(tag: str, arm: str):
    from teacher_sft.evaluate import compare
    runs.reload()
    result = compare(f"{ROOT}/{tag}/eval-base", f"{ROOT}/{tag}/eval-{arm}", f"{ROOT}/{tag}/compare-{arm}.json")
    runs.commit()
    print(result, flush=True)
    return result


@app.local_entrypoint()
def main(stage: str, tag: str = "glm53-sft-smoke", harvest_tag: str = "glm53-harvest",
         arm: str = "wins", max_candidates: int = 128, max_steps: int = 20, n_generated: int = 8,
         workers: int = 12, epochs: int = 1, max_target_tokens: int = 60000, rank: int = 8,
         learning_rate: float = 3e-5):
    import re
    for name in (tag, harvest_tag):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise ValueError("Tags must be simple names, not paths")
    if arm not in {"wins", "wins+corrections", "base"}:
        raise ValueError("arm must be wins, wins+corrections or base")
    if min(max_candidates, max_steps, n_generated) < 1:
        raise ValueError("Run bounds must be positive")
    if stage == "prepare":
        prepare.remote(harvest_tag, tag)
    elif stage == "author":
        revision = resolve_model.remote(tag)
        author.remote(tag, revision, max_candidates)
    elif stage == "train":
        if arm == "base":
            raise ValueError("Cannot train the base arm")
        train.remote(tag, arm, max_steps, epochs, max_target_tokens, rank, learning_rate)
    elif stage == "evaluate":
        evaluate.remote(tag, arm, n_generated, workers)
    elif stage == "compare":
        if arm == "base":
            raise ValueError("Choose the trained arm to compare")
        compare.remote(tag, arm)
    else:
        raise ValueError("stage must be prepare, author, train, evaluate or compare")
