"""v6 dynamic sampling: give dead base groups one bounded replacement attempt per iteration.

A group whose episodes all scored the same carries no gradient. Under v6's arm-2 win/loss reward,
"all rewards equal" is exactly "zero advantage"; under a shaped reward (the arm-3 repeat penalty) it
is not, which is why the driver refuses any other arm. A dead group's episodes are still banked; it
just does not count as a slot that delivered gradient.

Pure and deterministic: replacements are a function of the completed base episodes, the banks, the
stored depths and a seeded rng, and are persisted as replacements.json beside plans.json, so a
resumed iteration replays them instead of recomputing. Selection only; reward is never touched.
"""
from __future__ import annotations

import json
import os
import random
from typing import Callable

from redteamrl.train.allocation import (AllocParams, GroupPlan, WIN_LEVELS, depth_for, draw_restart,
                                        one_shot_depth, stored_depth, turn_for)
from redteamrl.train.banks import StartStateBank


def slot_episodes(episodes: dict[int, dict], plan: GroupPlan, n_rollouts: int) -> list[dict]:
    """The plan's episodes. Raises if any is missing: never plan from a partial base rollout."""
    ids = [plan.slot * n_rollouts + r for r in range(n_rollouts)]
    missing = [i for i in ids if i not in episodes]
    if missing:
        raise KeyError(f"slot {plan.slot} is missing episodes {missing}")
    return [episodes[i] for i in ids]


def is_dead(group: list[dict]) -> bool:
    """True iff every group_key bucket (the buckets assign_advantages uses; restart fallbacks form
    their own) has a single distinct reward."""
    buckets: dict[str, set] = {}
    for ep in group:
        buckets.setdefault(ep["group_key"], set()).add(ep["reward"])
    return all(len(rewards) == 1 for rewards in buckets.values())


def classify(episodes: dict[int, dict], plans: list[GroupPlan], n_rollouts: int) -> dict[int, bool]:
    return {p.slot: is_dead(slot_episodes(episodes, p, n_rollouts)) for p in plans}


def _key(task_id: str, entry_id: str, k: int, slot: int) -> str:
    return f"{task_id}@{entry_id}:{k}#slot{slot}"


def plan_replacements(episodes: dict[int, dict], base_plans: list[GroupPlan],
                      banks: dict[str, StartStateBank], depths: dict[str, float], n_slots: int,
                      n_rollouts: int, rng: random.Random, params: AllocParams) -> list[GroupPlan]:
    """Dead base groups in ascending slot order, at most `params.max_replacements` replacements.
    A dead restart retries its own entry at a one-shot corrected depth; a dead all-loss fresh group
    becomes a restart from its task's bank. Conceded slots (no different start state, all-win fresh,
    empty bank) do not use up the cap."""
    dead = classify(episodes, base_plans, n_rollouts)
    used = {p.entry.entry_id for p in base_plans if p.kind == "restart" and p.entry is not None}
    out: list[GroupPlan] = []
    for plan in sorted(base_plans, key=lambda p: p.slot):
        if len(out) >= params.max_replacements:
            break
        if not dead[plan.slot]:
            continue
        group = [ep for ep in slot_episodes(episodes, plan, n_rollouts)
                 if not ep.get("restart_fallback")]
        if not group:
            continue
        all_win = all(ep["won"] for ep in group)
        slot = n_slots + len(out)
        if plan.kind == "restart":
            entry = plan.entry
            ran_at = (plan.depth_frac if plan.depth_frac is not None
                      else stored_depth(entry.entry_id, depths, params))
            frac = one_shot_depth(ran_at, all_win, params)
            k = turn_for(frac, entry.key_turn)
            if k == plan.restart_turn:
                continue                    # clamped: no different start state to try
            direction = "earlier" if all_win else "later"
            out.append(GroupPlan(plan.task_id, "restart", slot,
                                 _key(plan.task_id, entry.entry_id, k, slot),
                                 f"replace{plan.slot}-{direction}", entry, k, frac))
        else:
            if all_win:
                continue                    # a restart is strictly easier: cannot make it live
            entry = draw_restart(banks.get(plan.task_id), rng, used)
            if entry is None:
                continue
            used.add(entry.entry_id)
            k = depth_for(entry, depths, params)
            kind = "win" if entry.level in WIN_LEVELS else "near-miss"
            out.append(GroupPlan(plan.task_id, "restart", slot,
                                 _key(plan.task_id, entry.entry_id, k, slot),
                                 f"replace{plan.slot}-{kind}", entry, k,
                                 stored_depth(entry.entry_id, depths, params)))
    return out


def load_or_plan(path: str, compute: Callable[[], list[GroupPlan]]) -> list[GroupPlan]:
    """The persisted replacement plan if one exists (resume), else compute and persist it."""
    if os.path.exists(path):
        with open(path) as handle:
            return [GroupPlan.from_dict(d) for d in json.load(handle)]
    plans = compute()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        json.dump([p.to_dict() for p in plans], handle)
    os.replace(path + ".tmp", path)
    return plans
