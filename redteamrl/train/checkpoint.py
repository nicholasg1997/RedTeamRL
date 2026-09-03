"""Torch-free checkpoint bookkeeping for resumable Modal training runs.

A checkpoint is a directory `iter{N}/` under a root on the Modal Volume. It is COMPLETE only
once its `meta.json` marker exists — the training script writes the adapter and optimizer state
first and `meta.json` LAST, so a dir that was cut off by a preemption mid-save is missing the
marker and is ignored here. Resume therefore never loads a half-written checkpoint.

Kept deliberately free of torch/peft so it is fully unit-tested offline; the model/optimizer
save+load lives in scripts/train_defender.py.
"""
import json
import os
import shutil

_META = "meta.json"


def write_meta(ckpt_dir: str, meta: dict) -> None:
    with open(os.path.join(ckpt_dir, _META), "w") as f:
        json.dump(meta, f)


def read_meta(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, _META)) as f:
        return json.load(f)


def validate_phase_identity(
    meta: dict,
    current: dict,
    checkpoint_path: str,
    *,
    allow_legacy: bool = False,
) -> None:
    """Reject an incompatible resume before model or optimizer state is loaded."""
    prior = meta.get("phase_identity")
    if prior is None:
        if allow_legacy:
            return
        raise RuntimeError(
            f"{checkpoint_path} has no phase_identity; refusing to resume an unverifiable "
            "legacy checkpoint. Set ALLOW_LEGACY_CHECKPOINT=True only after manually confirming "
            "that its model, adapter, opponent, and rollout recipe match."
        )
    if prior != current:
        raise RuntimeError(
            f"{checkpoint_path} holds a DIFFERENT phase; resuming would continue an incompatible "
            f"experiment.\n  on disk: {prior}\n  current: {current}\n"
            "Use a fresh CKPT_ROOT for a new alternating phase."
        )


def _complete(root: str) -> list[tuple[int, str]]:
    """(iter, path) for every COMPLETE checkpoint under root, ascending by iter."""
    if not os.path.isdir(root):
        return []
    out: list[tuple[int, str]] = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not (os.path.isdir(path) and os.path.isfile(os.path.join(path, _META))):
            continue
        try:
            it = int(read_meta(path)["iter"])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue
        out.append((it, path))
    return sorted(out)


def latest_checkpoint(root: str) -> str | None:
    """Path of the highest-iter complete checkpoint under root, or None if there is none."""
    comp = _complete(root)
    return comp[-1][1] if comp else None


def prune_checkpoints(root: str, keep: int) -> list[str]:
    """Delete all but the newest `keep` COMPLETE checkpoints (by iter). Incomplete dirs — e.g. a
    save still in flight — are never touched. Returns the deleted paths."""
    comp = _complete(root)
    to_delete = comp[:-keep] if keep > 0 else comp
    for _, path in to_delete:
        shutil.rmtree(path)
    return [path for _, path in to_delete]
