"""Deterministic per-episode canary variation for resumable SFT/evaluation runs."""

from __future__ import annotations

import base64
import hashlib

from redteamrl.types import SeedFile, TaskSpec


CANARY_SCHEME_VERSION = 2


def episode_canary(task_id: str, rollout_index: int, template: str, seed: int, revision: int) -> str:
	# The full value is high entropy. Keeping semantic prefixes such as ``sk-api`` would both make
	# memorization easier and create false fragment matches with benign values like ``payments-api``.
	material = f"v{CANARY_SCHEME_VERSION}:{revision}:{seed}:{task_id}:{rollout_index}:{template}"
	digest = hashlib.sha256(material.encode("utf-8")).digest()[:24]
	return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def randomize_task_canary(spec: TaskSpec, rollout_index: int, seed: int, revision: int) -> TaskSpec:
	"""Return an equivalent task whose planted canary is unique to this rollout.

	The derivation is deterministic, so Modal retries and base/candidate paired evaluations see
	the same value for a given task and rollout without storing random state.
	"""
	if len(spec.forbidden) != 1:
		raise ValueError("per-episode canary randomization requires exactly one forbidden value")

	old = spec.forbidden[0]
	new = episode_canary(spec.id, rollout_index, old, seed, revision)
	seed_files = [
		SeedFile(path=seed_file.path, content=seed_file.content.replace(old, new))
		for seed_file in spec.seed_files
	]
	if not any(new in seed_file.content for seed_file in seed_files):
		raise ValueError(f"task {spec.id} does not contain its forbidden canary in any seed file")
	return spec.model_copy(update={"forbidden": [new], "seed_files": seed_files}, deep=True)
