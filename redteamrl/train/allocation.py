"""Adaptive allocation: decide, once per iteration, what each GRPO group slot trains on.

A GRPO group only carries gradient when its outcomes differ. At G=8 and a ~6% win rate most tasks
produce all-loss groups, so full-length episodes on a task that never wins buy nothing. This module
fills a FIXED number of group slots (so token budgets stay comparable across lineages) with:

  * fresh groups  -- a normal episode from turn 0. A task gets one if it is live (it won recently),
                     brand new, or has gone `revisit_every` iterations without one, so a dead task
                     can come back to life. Leftover slots are filled by priority.
  * restart groups -- begin part-way through an earlier episode on the SAME task: a past win first,
                     else a near miss. Lowest-win-rate tasks first, capped at `restart_fraction_max`
                     of the slots so fresh episodes always stay at least half of an iteration.

Selection only: nothing here touches reward. `allocate` is a pure function of its arguments (the
caller seeds `rng` from the iteration), so a resumed iteration rebuilds the identical plan and its
banked episodes still line up with their groups.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass, field

from redteamrl.train.banks import BankEntry, StartStateBank

WIN_LEVELS = {3}
NEAR_MISS_LEVELS = {1, 2}


@dataclass(frozen=True)
class AllocParams:
    window: int = 3                      # iterations of fresh-start history behind a task's win rate
    revisit_every: int = 3               # a dead task is fresh-sampled at least this often
    restart_fraction_max: float = 0.5    # restart groups never exceed this share of the slots
    max_state_age_iters: int = 5         # bank entries older than this are evicted by the driver
    max_fresh_per_task: int = 2
    lambda_staleness: float = 0.05       # priority bonus per iteration without a fresh group
    prior_wins: float = 1.0              # smoothing: a task with 1 win in 8 is not "12.5% for sure"
    prior_losses: float = 9.0
    restart_depth_rule: str = "uniform_1_to_key"
    priority_version: str = "plr-v1"

    def as_identity(self) -> dict:
        return {f"alloc_{k}": v for k, v in self.__dict__.items()}


DEFAULT_PARAMS = AllocParams()


@dataclass
class TaskStats:
    """A task's recent FRESH-start record, persisted in the iteration checkpoint's meta."""
    task_id: str
    history: list[tuple[int, int]] = field(default_factory=list)   # (wins, episodes), newest last
    iters_since_fresh: int = 0

    @property
    def wins(self) -> int:
        return sum(w for w, _ in self.history)

    @property
    def episodes(self) -> int:
        return sum(n for _, n in self.history)

    def record(self, wins: int, episodes: int, window: int) -> None:
        """This iteration ran a fresh group for the task."""
        self.history = (self.history + [(wins, episodes)])[-window:]
        self.iters_since_fresh = 0

    def skip(self) -> None:
        """This iteration gave the task no fresh group."""
        self.iters_since_fresh += 1

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "history": [list(h) for h in self.history],
                "iters_since_fresh": self.iters_since_fresh}

    @classmethod
    def from_dict(cls, d: dict) -> "TaskStats":
        return cls(d["task_id"], [tuple(h) for h in d["history"]], d["iters_since_fresh"])


@dataclass
class GroupPlan:
    task_id: str
    kind: str                            # "fresh" | "restart"
    slot: int
    group_key: str                       # the GRPO baseline group (Example.group_key)
    reason: str                          # for the log: new / live / revisit / leftover / restart-*
    entry: BankEntry | None = None
    restart_turn: int | None = None      # number of prefix turns kept; the policy acts from here

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GroupPlan":
        entry = BankEntry(**d["entry"]) if d.get("entry") else None
        return cls(**{**d, "entry": entry})


def win_rate_estimate(stats: TaskStats, params: AllocParams = DEFAULT_PARAMS) -> float:
    return ((stats.wins + params.prior_wins)
            / (stats.episodes + params.prior_wins + params.prior_losses))


def task_priority(stats: TaskStats, params: AllocParams = DEFAULT_PARAMS) -> float:
    """How much a fresh group on this task is worth: learning potential p(1-p), the share of
    groups that carry gradient, plus a staleness bonus so a neglected task is eventually revisited
    (Prioritized Level Replay's two terms)."""
    p = win_rate_estimate(stats, params)
    return p * (1.0 - p) + params.lambda_staleness * stats.iters_since_fresh


def _due_reason(stats: TaskStats, params: AllocParams) -> str | None:
    if stats.episodes == 0:
        return "new"
    if stats.wins > 0:
        return "live"
    if stats.iters_since_fresh >= params.revisit_every:
        return "revisit"
    return None


def _draw_restart(bank: StartStateBank | None, rng: random.Random, used: set[str]):
    if bank is None:
        return None
    return (bank.sample(rng, levels=WIN_LEVELS, exclude=used)
            or bank.sample(rng, levels=NEAR_MISS_LEVELS, exclude=used))


def allocate(task_stats: list[TaskStats], banks: dict[str, StartStateBank], n_slots: int,
             iteration: int, rng: random.Random,
             params: AllocParams = DEFAULT_PARAMS) -> list[GroupPlan]:
    stats = sorted(task_stats, key=lambda s: s.task_id)
    max_restarts = int(params.restart_fraction_max * n_slots)

    # Restarts: one per task per pass, lowest estimated win rate first; each bank entry is used at
    # most once this iteration.
    restarts: list[tuple[str, BankEntry, int]] = []
    used: set[str] = set()
    low_first = sorted(stats, key=lambda s: (win_rate_estimate(s, params), s.task_id))
    while len(restarts) < max_restarts:
        progressed = False
        for s in low_first:
            if len(restarts) >= max_restarts:
                break
            entry = _draw_restart(banks.get(s.task_id), rng, used)
            if entry is None:
                continue
            used.add(entry.entry_id)
            restarts.append((s.task_id, entry, rng.randint(1, entry.key_turn)))
            progressed = True
        if not progressed:
            break

    # Fresh groups fill the rest: due tasks by priority, then leftovers by priority.
    n_fresh = n_slots - len(restarts)
    by_priority = sorted(stats, key=lambda s: (-task_priority(s, params), s.task_id))
    fresh: list[tuple[str, str]] = []                      # (task_id, reason)
    # New and revisit-due tasks go BEFORE live ones: the revisit is the guarantee that a dead task
    # is sampled again, and a priority comparison alone would let live tasks crowd it out forever.
    due = [(s.task_id, r) for s in by_priority if (r := _due_reason(s, params))]
    due.sort(key=lambda tr: tr[1] == "live")               # stable: priority order within each group
    fresh.extend(due[:n_fresh])
    counts = Counter(t for t, _ in fresh)
    while len(fresh) < n_fresh:
        added = False
        for s in by_priority:
            if len(fresh) < n_fresh and counts[s.task_id] < params.max_fresh_per_task:
                fresh.append((s.task_id, "leftover"))
                counts[s.task_id] += 1
                added = True
        if not added:
            raise ValueError(f"cannot fill {n_slots} slots: {len(stats)} tasks, "
                             f"max_fresh_per_task={params.max_fresh_per_task}, "
                             f"{len(restarts)} restarts")

    plans: list[GroupPlan] = []
    for task_id, reason in sorted(fresh):
        slot = len(plans)
        plans.append(GroupPlan(task_id, "fresh", slot, f"{task_id}#slot{slot}", reason))
    for task_id, entry, k in restarts:
        slot = len(plans)
        kind = "win" if entry.level in WIN_LEVELS else "near-miss"
        plans.append(GroupPlan(task_id, "restart", slot,
                               f"{task_id}@{entry.entry_id}:{k}#slot{slot}",
                               f"restart-{kind}", entry, k))
    return plans
