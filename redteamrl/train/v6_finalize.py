"""End-of-iteration bookkeeping for lineage v6 (dynamic sampling, revisit backoff, adaptive depth).

Wraps v5's finalize_iteration over base + replacement episodes and adds:
  * dead-group accounting: dead_before (base slots), replacements, replacements_live, dead_after;
  * the backoff rule: an unwon fresh REVISIT grows a task's dead_revisits, a fresh win resets it,
    a restart win never does (the backoff tracks fresh-start capability);
  * the finalize-time depth update for every restart entry that ran, returned as a new map.

Pure over the persisted episode store; nothing reads or changes reward.
"""
from __future__ import annotations

from redteamrl.train.allocation import AllocParams, GroupPlan, TaskStats, stored_depth, update_depths
from redteamrl.train.banks import StartStateBank
from redteamrl.train.replacement import classify, slot_episodes
from redteamrl.train.v5_finalize import DEFAULT_MAX_NEAR_MISSES, finalize_iteration, v5_identity


def _usable(episodes: dict[int, dict], plan: GroupPlan, n_rollouts: int) -> list[dict]:
    return [ep for ep in slot_episodes(episodes, plan, n_rollouts) if not ep.get("restart_fallback")]


def _bucket(plan: GroupPlan) -> str:
    frac = plan.restart_turn / plan.entry.key_turn
    return "lo" if frac < 1 / 3 else "mid" if frac < 2 / 3 else "hi"


def finalize_v6(episodes: dict[int, dict], base_plans: list[GroupPlan],
                replacement_plans: list[GroupPlan], banks: dict[str, StartStateBank],
                stats: dict[str, TaskStats], depths: dict[str, float], iteration: int,
                n_rollouts: int, record_dir: str, params: AllocParams,
                max_near_misses: int = DEFAULT_MAX_NEAR_MISSES,
                update_stats: bool = True) -> tuple[dict, dict[str, float]]:
    all_plans = list(base_plans) + list(replacement_plans)
    summary = finalize_iteration(episodes, all_plans, banks, stats, iteration, n_rollouts,
                                 record_dir, params, max_near_misses=max_near_misses,
                                 update_stats=update_stats)

    if update_stats:
        fresh_by_task: dict[str, list[GroupPlan]] = {}
        for plan in base_plans:
            if plan.kind == "fresh":
                fresh_by_task.setdefault(plan.task_id, []).append(plan)
        for task_id, plans in fresh_by_task.items():
            won_any = any(ep["won"] for p in plans for ep in _usable(episodes, p, n_rollouts))
            stats[task_id].note_fresh_outcome(won_any=won_any,
                                              revisited=any(p.reason == "revisit" for p in plans))

    restarts = sorted((p for p in all_plans if p.kind == "restart" and p.entry is not None),
                      key=lambda p: p.slot)
    groups, fracs, by_bucket = [], [], {}
    for plan in restarts:
        ran_at = (plan.depth_frac if plan.depth_frac is not None
                  else stored_depth(plan.entry.entry_id, depths, params))
        fracs.append(ran_at)
        usable = _usable(episodes, plan, n_rollouts)
        if not usable:
            continue
        groups.append((plan.entry.entry_id, ran_at, sum(ep["won"] for ep in usable) / len(usable)))
        by_bucket.setdefault(_bucket(plan), []).extend(bool(ep["won"]) for ep in usable)
    live_ids = {e.entry_id for bank in banks.values() for e in bank.entries()}
    new_depths = update_depths(depths, groups, live_ids, params)

    dead = classify(episodes, all_plans, n_rollouts)
    dead_before = sum(dead[p.slot] for p in base_plans)
    replacements_live = sum(not dead[p.slot] for p in replacement_plans)
    summary.update({
        "n_fresh_groups": sum(p.kind == "fresh" for p in base_plans),
        "n_restart_groups": sum(p.kind == "restart" for p in base_plans),
        "n_base_groups": len(base_plans),
        "dead_before": dead_before,
        "replacements": len(replacement_plans),
        "replacements_live": replacements_live,
        "dead_after": dead_before - replacements_live,
        "depth_frac_mean": sum(fracs) / len(fracs) if fracs else None,
        "win_by_depth": {b: sum(w) / len(w) for b, w in sorted(by_bucket.items())},
        "backed_off": {t: s.dead_revisits for t, s in sorted(stats.items()) if s.dead_revisits > 0},
    })
    return summary, new_depths


def v6_identity(params: AllocParams, max_near_misses: int = DEFAULT_MAX_NEAR_MISSES) -> dict:
    return {**v5_identity(params, max_near_misses), "dead_rule": "uniform-outcome",
            "replacement_version": 1}
