"""End-of-iteration bookkeeping for lineage v5 (adaptive allocation + restarts).

Pure functions over the episode store's payloads (`train/episode_store.read_episodes`), which hold
everything needed and survive a mid-iteration resume:

  * per-task FRESH-start stats (the input to the next iteration's allocation) -- restart groups
    never count, since they begin from a favourable state by construction;
  * what enters the start-state banks: every win, plus a capped number of the best near misses;
  * eviction of bank entries past `max_state_age_iters`, and of the records they point at;
  * the fresh-vs-restart summary for the log. The restart win rate is NEVER a headline number.

Nothing here reads or changes reward.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

from redteamrl.harness.restart import save_record
from redteamrl.train.allocation import AllocParams, GroupPlan, TaskStats
from redteamrl.train.banks import BankEntry, StartStateBank, route_signature

DEFAULT_MAX_NEAR_MISSES = 2


def _is_fresh(ep: dict) -> bool:
    """Ran from turn 0: a fresh group, or a restart rollout that fell back to a fresh episode."""
    return ep.get("kind") == "fresh" or bool(ep.get("restart_fallback"))


def _bank_one(banks: dict[str, StartStateBank], ep: dict, level: int, key_turn: int,
              iteration: int, record_dir: str) -> bool:
    task_id = ep["task_id"]
    entry_id = f"{task_id}-i{iteration}-e{ep['episode_id']}"
    path = os.path.join(record_dir, f"{entry_id}.json")
    signature = route_signature(SimpleNamespace(steps=ep["record"]["steps"]), upto_turn=key_turn)
    entry = BankEntry(entry_id, task_id, iteration, level, key_turn, signature, path)
    if not banks[task_id].add(entry):
        return False
    save_record(path, ep["record"])
    return True


def bank_episodes(episodes: dict[int, dict], banks: dict[str, StartStateBank], iteration: int,
                  record_dir: str, max_near_misses: int = DEFAULT_MAX_NEAR_MISSES,
                  params: AllocParams | None = None) -> tuple[int, int]:
    """Bank every win and each task's best fresh near misses; returns (wins, near misses) banked.

    Idempotent: an entry id is a function of (task, iteration, episode), so re-banking the same
    episodes rewrites the same entries. v7 relies on this to bank the base pass BEFORE replacement
    planning and again at finalize.

    `params.near_miss_offset` moves a near miss's restart point past its redaction, so the restart
    begins with the redacted result in context; one leaving fewer than
    `params.near_miss_min_turns_left` turns after that point is not banked. Wins are unaffected."""
    offset = params.near_miss_offset if params is not None else 0
    min_left = params.near_miss_min_turns_left if params is not None else 0
    banked_wins = banked_near = 0
    near_by_task: dict[str, list[tuple[int, int, dict]]] = {}
    for ep in episodes.values():
        if "record" not in ep:
            continue
        level, key_turn = ep["closeness"]
        if key_turn < 1:
            continue                     # nothing before it to restart from; takes no quota
        if level == 3:
            banked_wins += _bank_one(banks, ep, level, key_turn, iteration, record_dir)
        elif level in (1, 2) and _is_fresh(ep):
            # A restart inherits its source's prefix, so its near miss is usually the source's own
            # (already banked). Only fresh episodes add new near-miss starts; a restart's WIN is
            # always new, since the prefix stops before the source's key turn.
            start = key_turn + offset
            if len(ep["record"]["steps"]) - start < min_left:
                continue                 # too few turns left after the redaction to teach anything
            near_by_task.setdefault(ep["task_id"], []).append((level, start, ep))
    for task_id, candidates in near_by_task.items():
        # Closest first, then earliest (most turns left to find its own way past the defender);
        # episode id breaks ties so the choice is deterministic.
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]["episode_id"]))
        for level, key_turn, ep in candidates[:max_near_misses]:
            banked_near += _bank_one(banks, ep, level, key_turn, iteration, record_dir)
    return banked_wins, banked_near


def finalize_iteration(episodes: dict[int, dict], plans: list[GroupPlan],
                       banks: dict[str, StartStateBank], stats: dict[str, TaskStats],
                       iteration: int, n_rollouts: int, record_dir: str, params: AllocParams,
                       max_near_misses: int = DEFAULT_MAX_NEAR_MISSES,
                       update_stats: bool = True) -> dict:
    """Update stats and banks from one finished iteration. Banking is idempotent (entry ids are a
    function of task, iteration and episode), so a resume that re-runs it does not double up.
    `update_stats=False` skips the stats, which the driver has already written to the checkpoint."""
    fresh_by_task: dict[str, list[dict]] = {}
    for plan in plans:
        if plan.kind != "fresh":
            continue
        fresh_by_task.setdefault(plan.task_id, []).extend(
            episodes[plan.slot * n_rollouts + r] for r in range(n_rollouts)
            if plan.slot * n_rollouts + r in episodes)
    if update_stats:
        for task_id, task_stats in stats.items():
            group = fresh_by_task.get(task_id)
            if group:
                task_stats.record(sum(ep["won"] for ep in group), len(group), params.window)
            else:
                task_stats.skip()

    banked_wins, banked_near = bank_episodes(episodes, banks, iteration, record_dir,
                                             max_near_misses, params=params)

    evicted = 0
    for bank in banks.values():
        for entry in bank.evict_older_than(iteration, params.max_state_age_iters):
            evicted += 1
            try:
                os.remove(entry.record_path)
            except OSError:
                pass
    summary = summarize(episodes, plans, n_rollouts)
    summary.update(banked_wins=banked_wins, banked_near=banked_near, evicted=evicted)
    return summary


def summarize(episodes: dict[int, dict], plans: list[GroupPlan], n_rollouts: int) -> dict:
    fresh = [e for e in episodes.values() if _is_fresh(e)]
    restart = [e for e in episodes.values() if not _is_fresh(e)]
    by_task: dict[str, list[bool]] = {}
    for e in fresh:
        by_task.setdefault(e["task_id"], []).append(bool(e["won"]))
    task_rates = [sum(v) / len(v) for v in by_task.values()]
    rate = lambda eps: (sum(bool(e["won"]) for e in eps) / len(eps)) if eps else 0.0
    return {
        "fresh_n": len(fresh), "restart_n": len(restart),
        "fallbacks": sum(1 for e in episodes.values() if e.get("restart_fallback")),
        "fresh_win_rate": rate(fresh),
        "fresh_win_rate_task_avg": (sum(task_rates) / len(task_rates)) if task_rates else 0.0,
        "restart_win_rate": rate(restart),
        "n_fresh_groups": sum(p.kind == "fresh" for p in plans),
        "n_restart_groups": sum(p.kind == "restart" for p in plans),
    }


def v5_identity(params: AllocParams, max_near_misses: int = DEFAULT_MAX_NEAR_MISSES) -> dict:
    """The v5 settings that change the episode distribution, for ``phase_identity``: resuming a
    checkpoint across a change to any of them would silently mix two different training regimes."""
    return {**params.as_identity(), "alloc_max_near_misses": max_near_misses,
            "restart_state_version": 1}
