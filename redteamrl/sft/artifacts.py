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
        "counterfactual_probe_variants",
        "max_probe_regression",
        # Legacy round manifests used a benign-improvement objective. Keep it eval-only so moving
        # the gate to attack-side improvement does not invalidate upstream banks.
        "min_benign_leak_improvement",
        "min_attack_policy_improvement",
        "collapse_tolerance",
        "blind_review_episodes",
        "max_utility_regression",
    }
)

# These fields change sampling, target rendering, balancing, or gradient training, but cannot change
# an already collected episode or an already authored rationalization. Keeping this tier separate
# lets a run reuse the expensive upstream banks while creating content-addressed downstream
# artifacts for a new training recipe.
TRAINING_ONLY_MANIFEST_KEYS = frozenset(
    {
        "training_revision",
        "sft_epochs",
        "sft_lr",
        "sft_batch",
        "sft_microbatch",
        "max_call_pair_ratio",
        "redact_to_pass_ratio",
        "duplicate_pass_prompt_cap",
        "min_class_examples",
        "min_authored_redact_examples",
        "decision_loss_weight",
        "anchor_to_correction_ratio",
        "min_anchors_per_stratum",
        "max_sft_steps",
        "lora_r",
        "lora_alpha",
        "lora_target_modules",
        "counterfactual_version",
        "counterfactual_train_variants",
        "target_render_version",
    }
)

# These settings change which collected records are rationalized, or how the rationalizer/filter
# processes them, but they do not change the episodes themselves. They therefore select a new
# content-addressed rationalization bank while preserving the expensive episode bank.
RATIONALIZATION_ONLY_MANIFEST_KEYS = frozenset(
    {
        "rationalization_revision",
        "redact_rationalization_attempts",
        "review_holdout_fraction",
        "review_split_version",
    }
)


def episode_manifest_identity(manifest: dict) -> dict:
    """Return fields whose change can make collected episodes stale."""
    return {
        key: value
        for key, value in manifest.items()
        if key
        not in (
            EVAL_ONLY_MANIFEST_KEYS
            | TRAINING_ONLY_MANIFEST_KEYS
            | RATIONALIZATION_ONLY_MANIFEST_KEYS
        )
    }


def training_manifest_identity(manifest: dict) -> dict:
    """Return fields whose change can make a trained adapter stale."""
    return {
        key: value
        for key, value in manifest.items()
        if key not in EVAL_ONLY_MANIFEST_KEYS
    }


def rationalization_manifest_identity(manifest: dict) -> dict:
    """Return fields whose change can make authored rationalizations stale."""
    return {
        key: value
        for key, value in manifest.items()
        if key not in EVAL_ONLY_MANIFEST_KEYS | TRAINING_ONLY_MANIFEST_KEYS
    }


def rationalization_artifact_tag(manifest: dict) -> str:
    """Stable fingerprint for the selected records and authored/filter output."""
    payload = json.dumps(
        rationalization_manifest_identity(manifest),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def training_artifact_tag(manifest: dict) -> str:
    """Stable fingerprint for examples, balancing, optimizer settings, and the adapter."""
    payload = json.dumps(
        training_manifest_identity(manifest),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def evaluation_artifact_tag(tag: str, config: dict) -> str:
    """Add a stable config fingerprint so incompatible eval checkpoints never mix."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"{tag}-{digest}"
