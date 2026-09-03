"""Deterministic held-out split for the blinded self-review corpus.

The review measurement is only evidence of generalization if the reviewed episodes never entered
rationalization or SFT. This partitions the ALREADY-BANKED episode keys, so no re-collection is
needed -- the reserved share is simply skipped downstream.

Membership is a pure function of the key, not of the corpus: a resumed run that enumerates a
different number of banked episodes must not reshuffle the split and quietly pull a trained-on
episode into the held-out set.
"""

from __future__ import annotations

import hashlib

REVIEW_SPLIT_VERSION = 1
_BUCKETS = 10_000


def is_review_held_out(episode_key: str, fraction: float = 0.2, seed: int = 0) -> bool:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    material = f"review-split-v{REVIEW_SPLIT_VERSION}:{seed}:{episode_key}".encode()
    bucket = int.from_bytes(hashlib.sha256(material).digest()[:4], "big") % _BUCKETS
    return bucket < fraction * _BUCKETS


def split_episode_keys(
    episode_keys: list[str], fraction: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Return (train_keys, review_keys). Order-preserving and disjoint."""
    train, review = [], []
    for key in episode_keys:
        (review if is_review_held_out(key, fraction, seed) else train).append(key)
    return train, review
