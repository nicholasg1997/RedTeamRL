"""Per-episode rollout persistence so a preemption costs one episode, not one iteration.

Checkpointing only between iterations means a preemption at task 6/9 discards ~3 hours of
completed episodes and restarts the iteration from scratch. This is the same fix
``sft/collect.py`` already applies to collection: write one atomic record per finished episode,
and skip banked work on resume.

Resumed episodes stay valid because the policy does not move during a rollout — the adapter is
restored from the previous iteration's checkpoint, which is exactly what generated them. A record
is only readable once its final rename lands, so a torn write is invisible to resume.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict

from redteamrl.train.capture import Example


_FINAL = re.compile(r"^(\d{6})\.json$")


def _path(store_dir: str, episode_id: int) -> str:
    return os.path.join(store_dir, f"{episode_id:06d}.json")


def write_episode(store_dir: str, payload: dict) -> None:
    """Persist one finished episode atomically. Safe to call from worker threads: every episode
    owns a distinct filename, and the rename is what publishes it."""
    os.makedirs(store_dir, exist_ok=True)
    serializable = dict(payload)
    serializable["examples"] = [asdict(example) for example in payload["examples"]]
    final = _path(store_dir, int(payload["episode_id"]))
    tmp = final + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(serializable, handle)
    os.replace(tmp, final)


def banked_episode_ids(store_dir: str) -> set[int]:
    """Episode ids with a COMPLETE record on disk. Ignores in-flight .tmp files."""
    if not os.path.isdir(store_dir):
        return set()
    return {
        int(match.group(1))
        for name in os.listdir(store_dir)
        if (match := _FINAL.match(name))
    }


def read_episodes(store_dir: str) -> dict[int, dict]:
    """Load every complete record, rehydrating Examples. A corrupt final file is skipped rather
    than aborting the run: losing one episode is cheaper than losing the iteration."""
    if not os.path.isdir(store_dir):
        return {}
    episodes: dict[int, dict] = {}
    for episode_id in sorted(banked_episode_ids(store_dir)):
        try:
            with open(_path(store_dir, episode_id)) as handle:
                payload = json.load(handle)
            payload["examples"] = [Example(**row) for row in payload["examples"]]
        except (json.JSONDecodeError, TypeError, KeyError, OSError):
            continue
        episodes[episode_id] = payload
    return episodes
