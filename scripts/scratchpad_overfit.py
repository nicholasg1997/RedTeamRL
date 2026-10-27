"""Small train-only learning/serving control. No teacher calls and no shell rollouts.

First: modal run scripts/scratchpad_overfit.py --stage prepare --tag overfit-8b-v1
Then:  modal run --detach scripts/scratchpad_overfit.py --stage run --tag overfit-8b-v1
"""
import modal

ROOT = "/runs/overfit-control"
SOURCE_ROOT = "/runs/teacher-sft"
image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.21.0", "requests", "torch>=2.2", "transformers<5", "peft>=0.11",
                 "accelerate>=0.30", "pydantic", "pyyaml", "tqdm")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("redteamrl", remote_path="/root/redteamrl")
    .add_local_dir("teacher_sft", remote_path="/root/teacher_sft")
    .add_local_dir("overfit_probe", remote_path="/root/overfit_probe"))
runs = modal.Volume.from_name("redteamrl-eval-runs", create_if_missing=True)
cache = modal.Volume.from_name("redteamrl-hf-cache", create_if_missing=True)
volumes = {"/runs": runs, "/cache": cache}
app = modal.App("redteamrl-overfit-control", image=image)


def source_identity():
    from redteamrl.scratchpad.experiment import source_digest
    from teacher_sft.data import digest
    return digest([source_digest(f"/root/{package}", __file__)
                   for package in ("redteamrl", "teacher_sft", "overfit_probe")])


@app.function(cpu=2, timeout=3600, volumes=volumes)
def prepare(tag: str = "overfit-8b-v1", source_tag: str = "warmstart-v1", pairs: int = 4):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from teacher_sft.author import load_examples
    from overfit_probe.data import prepare
    runs.reload()
    cache.reload()
    source = f"{SOURCE_ROOT}/{source_tag}"
    meta, _ = load_examples(f"{source}/authored")
    model, revision = meta["config"]["model"], meta["config"]["revision"]
    if model not in {"Qwen/Qwen3-8B", "Qwen/Qwen3-14B"}:
        raise ValueError("This control is validated for Qwen3-8B/14B data only")
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    report = prepare(source, f"{ROOT}/{tag}", tokenizer,
        dict(source=source_identity(), source_tag=source_tag, pairs=pairs,
             max_length=8192, max_new_tokens=3072), commit=runs.commit)
    # Transfer weights on CPU, only after the data selection succeeds.
    snapshot_download(model, revision=revision,
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    cache.commit()
    print(report, flush=True)
    return report


@app.function(gpu="A100-80GB", cpu=8, timeout=3600, volumes=volumes)
def run(tag: str = "overfit-8b-v1", max_steps: int = 40, max_target_tokens: int = 80000):
    from overfit_probe.data import load_cases
    from overfit_probe.runner import (DEFAULT_CONFIG, validate_config, train_and_probe,
                                      served_probes, decision_report)
    runs.reload()
    cache.reload()
    root = f"{ROOT}/{tag}"
    selected, _ = load_cases(root)
    source = source_identity()
    if selected["config"]["source"] != source:
        raise ValueError("Source changed since selection; prepare a fresh tag")
    config = {**DEFAULT_CONFIG, "source": source, "max_steps": max_steps,
              "max_target_tokens": max_target_tokens}
    validate_config(config)
    fit = train_and_probe(root, config, commit=runs.commit)
    served_probes(root, fit, commit=runs.commit)
    result = decision_report(root)
    runs.commit()
    print(result, flush=True)
    return result


@app.local_entrypoint()
def main(stage: str = "prepare", tag: str = "overfit-8b-v1", source_tag: str = "warmstart-v1",
         pairs: int = 4, max_steps: int = 40, max_target_tokens: int = 80000):
    import re
    from overfit_probe.runner import DEFAULT_CONFIG, validate_config
    for name in (tag, source_tag):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise ValueError("Tags must be simple names, not paths")
    if not 2 <= pairs <= 8:
        raise ValueError("pairs must be 2-8")
    validate_config({**DEFAULT_CONFIG, "max_steps": max_steps, "max_target_tokens": max_target_tokens})
    if stage == "prepare":
        prepare.remote(tag, source_tag, pairs)
    elif stage == "run":
        run.remote(tag, max_steps, max_target_tokens)
    else:
        raise ValueError("stage must be prepare or run")
