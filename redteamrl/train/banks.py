"""Per-task start-state banks: the earlier episodes a restart group can begin from.

A bank holds POINTERS (BankEntry -> the persisted source record from harness/restart.py), one JSON
file each, written atomically like train/episode_store.py so a preemption never leaves a torn entry.
`directory=None` keeps a bank in memory (tests).

Sampling is uniform over distinct ROUTES first, then over entries within a route. That is the
collapse guard: a route that has won 30 times is no likelier to seed a restart than one that has
won once, so restarts keep teaching the rare routes instead of cementing the common one.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BankEntry:
    entry_id: str
    task_id: str
    iteration: int
    level: int            # closeness level of the source (3 = win)
    key_turn: int         # 0-indexed turn the source reached `level`
    route_signature: str
    record_path: str      # persisted source record (harness/restart.py)


def route_signature(result, upto_turn: int | None = None) -> str:
    """Coarse, wording-free summary of the action path, up to and including `upto_turn`.

    Per turn: the action kind, the tool, the file path argument, and for shell commands only the
    FIRST WORD (cat / base64 / grep ...), which marks the tactic without making every phrasing
    distinct. Consecutive identical turns collapse, so retry loops do not create new routes."""
    parts: list[tuple] = []
    for step in result.steps:
        if upto_turn is not None and step["turn"] > upto_turn:
            break
        action = step["action"]
        args = action.get("args") or {}
        command = (args.get("command") or "").split()
        part = (action.get("kind"), action.get("tool"), args.get("path"),
                command[0] if command else None)
        if not parts or parts[-1] != part:
            parts.append(part)
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:12]


class StartStateBank:
    def __init__(self, directory: str | None):
        self.directory = directory
        self._entries: dict[str, BankEntry] = {}
        if directory and os.path.isdir(directory):
            for name in sorted(os.listdir(directory)):
                if not name.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(directory, name)) as handle:
                        entry = BankEntry(**json.load(handle))
                except (json.JSONDecodeError, TypeError, OSError):
                    continue           # one torn/foreign file must not lose the bank
                self._entries[entry.entry_id] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[BankEntry]:
        return [self._entries[k] for k in sorted(self._entries)]

    def add(self, entry: BankEntry) -> bool:
        """Bank `entry`. A key turn below 1 is rejected: a restart keeps at least one prefix turn
        (spec s6.3), so a source that peaked on its first turn has nothing to restart from."""
        if entry.key_turn < 1:
            return False
        self._entries[entry.entry_id] = entry
        if self.directory:
            os.makedirs(self.directory, exist_ok=True)
            final = os.path.join(self.directory, f"{entry.entry_id}.json")
            with open(final + ".tmp", "w") as handle:
                json.dump(asdict(entry), handle)
            os.replace(final + ".tmp", final)
        return True

    def evict_older_than(self, current_iter: int, max_age: int) -> int:
        stale = [e for e in self._entries.values() if current_iter - e.iteration > max_age]
        for entry in stale:
            del self._entries[entry.entry_id]
            if self.directory:
                try:
                    os.remove(os.path.join(self.directory, f"{entry.entry_id}.json"))
                except OSError:
                    pass
        return len(stale)

    def distinct_routes(self, levels: set[int] | None = None) -> int:
        return len({e.route_signature for e in self._eligible(levels)})

    def _eligible(self, levels: set[int] | None) -> list[BankEntry]:
        return [e for e in self.entries() if levels is None or e.level in levels]

    def sample(self, rng: random.Random, levels: set[int] | None = None,
               exclude: set[str] | frozenset = frozenset()) -> BankEntry | None:
        """One entry: a uniformly chosen route, then a uniformly chosen entry on it. Entries are
        visited in sorted order so the draw is a pure function of the rng state (resume).
        `exclude` removes entry ids already used this iteration (the per-entry reuse cap)."""
        pool = [e for e in self._eligible(levels) if e.entry_id not in exclude]
        if not pool:
            return None
        routes = sorted({e.route_signature for e in pool})
        route = routes[rng.randrange(len(routes))]
        on_route = [e for e in pool if e.route_signature == route]
        return on_route[rng.randrange(len(on_route))]
