"""End-of-iteration bookkeeping for lineage v7: v6's finalize plus graduation.

A restart entry GRADUATES when its group comes back all-win at the earliest start it can reach
(`allocation.can_go_earlier` is False: k = 1, or clamped at depth_min). The adaptive depth rule
would answer "too easy" by starting earlier, and there is nowhere earlier to go, so the entry has
nothing left to teach: it leaves restart sampling. Its route still counts in the bank's diversity,
and its TASK stays in fresh rotation -- graduation retires a start state, never a task.

Also counts replacements drawn from this iteration's own banked episodes (same-iteration banking),
so v7 can be compared with v6 on exactly that. Selection only; nothing reads or changes reward.
"""
from __future__ import annotations

from redteamrl.train.allocation import AllocParams, GroupPlan, TaskStats, can_go_earlier
from redteamrl.train.banks import StartStateBank
from redteamrl.train.replacement import classify, slot_episodes
from redteamrl.train.v5_finalize import DEFAULT_MAX_NEAR_MISSES
from redteamrl.train.v6_finalize import finalize_v6, v6_identity


def finalize_v7(episodes: dict[int, dict], base_plans: list[GroupPlan],
                replacement_plans: list[GroupPlan], banks: dict[str, StartStateBank],
                stats: dict[str, TaskStats], depths: dict[str, float], graduated: set[str],
                iteration: int, n_rollouts: int, record_dir: str, params: AllocParams,
                max_near_misses: int = DEFAULT_MAX_NEAR_MISSES,
                update_stats: bool = True) -> tuple[dict, dict[str, float], set[str]]:
    summary, new_depths = finalize_v6(episodes, base_plans, replacement_plans, banks, stats,
                                      depths, iteration, n_rollouts, record_dir, params,
                                      max_near_misses=max_near_misses, update_stats=update_stats)
    new = set()
    for plan in list(base_plans) + list(replacement_plans):
        if plan.kind != "restart" or plan.entry is None:
            continue
        usable = [ep for ep in slot_episodes(episodes, plan, n_rollouts)
                  if not ep.get("restart_fallback")]
        if usable and all(ep["won"] for ep in usable) and not can_go_earlier(plan, params):
            new.add(plan.entry.entry_id)
    live_ids = {e.entry_id for bank in banks.values() for e in bank.entries()}
    graduated_now = (set(graduated) | new) & live_ids      # evicted entries drop out

    dead = classify(episodes, replacement_plans, n_rollouts) if replacement_plans else {}
    same_iter = [p for p in replacement_plans
                 if p.entry is not None and p.entry.iteration == iteration]
    summary.update({
        "graduated": len(new - set(graduated)),
        "graduated_total": len(graduated_now),
        "same_iter_replacements": len(same_iter),
        "same_iter_replacements_live": sum(not dead[p.slot] for p in same_iter),
    })
    return summary, new_depths, graduated_now


def v7_identity(params: AllocParams, max_near_misses: int = DEFAULT_MAX_NEAR_MISSES) -> dict:
    return {**v6_identity(params, max_near_misses), "graduation": "all-win-at-floor-v1",
            "same_iter_banking": 1}
