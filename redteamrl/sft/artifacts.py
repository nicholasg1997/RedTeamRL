"""Run-identity helpers for resumable SFT and evaluation artifacts."""

from __future__ import annotations

import hashlib
import json


# These settings can change which evaluation artifacts are produced, but cannot change a collected
# episode, authored SFT example, or trained adapter. They therefore must not invalidate banked
# training work. Evaluation artifacts get their own content-derived identity below.
EVAL_ONLY_MANIFEST_KEYS = frozenset(
    {
        "evaluation_backend",
        "eval_rollouts",
        "train_eval_rollouts",
        "eval_checkpoint_every",
        "eval_defender_temperature",
        "min_benign_leak_improvement",
        "max_utility_regression",
    }
)


def training_manifest_identity(manifest: dict) -> dict:
    """Return only fields whose change can make collected or trained data stale."""
    return {
        key: value
        for key, value in manifest.items()
        if key not in EVAL_ONLY_MANIFEST_KEYS
    }


def evaluation_artifact_tag(tag: str, config: dict) -> str:
    """Add a stable config fingerprint so incompatible eval checkpoints never mix."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"{tag}-{digest}"
