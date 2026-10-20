"""End-of-iteration bookkeeping for lineage v5: per-task stats, what enters the banks, eviction,
and the fresh-vs-restart summary. Pure functions over the episode store's payloads."""
import json
import os

from redteamrl.train.allocation import DEFAULT_PARAMS, GroupPlan, TaskStats
from redteamrl.train.banks import StartStateBank
from redteamrl.train.v5_finalize import finalize_iteration, summarize

N = 4          # rollouts per group
P = DEFAULT_PARAMS


def _record(tag="a", n_steps=6):
    steps = [{"turn": t, "action": {"kind": "tool_call", "tool": "read_file",
                                    "args": {"path": f"/{tag}{t}"}}} for t in range(n_steps)]
    return {"version": 1, "steps": steps, "spec": {}, "raw_outputs": [], "fs_hash_after": []}


def _ep(plan, r, level, key_turn, won=None, kind=None, tag="a", fallback=False):
    return {"episode_id": plan.slot * N + r, "task_id": plan.task_id,
            "kind": kind or plan.kind, "slot": plan.slot,
            "won": (level == 3) if won is None else won,
            "closeness": [level, key_turn], "record": _record(tag),
            "restart_fallback": fallback}


def _fresh(task, slot):
    return GroupPlan(task, "fresh", slot, f"{task}#slot{slot}", "live")


def _restart(task, slot):
    return GroupPlan(task, "restart", slot, f"{task}@e:{2}#slot{slot}", "restart-win")


def _index(eps):
    return {e["episode_id"]: e for e in eps}


def _setup(tmp_path, tasks=("A", "B")):
    banks = {t: StartStateBank(str(tmp_path / "banks" / t)) for t in tasks}
    stats = {t: TaskStats(t) for t in tasks}
    return banks, stats, str(tmp_path / "records")


def test_stats_record_fresh_wins_and_skip_the_rest(tmp_path):
    banks, stats, rec = _setup(tmp_path)
    plans = [_fresh("A", 0)]
    eps = _index([_ep(plans[0], r, 3 if r == 0 else 0, 4) for r in range(N)])
    finalize_iteration(eps, plans, banks, stats, 1, N, rec, P)
    assert stats["A"].history == [(1, N)] and stats["A"].iters_since_fresh == 0
    assert stats["B"].history == [] and stats["B"].iters_since_fresh == 1


def test_restart_episodes_never_count_toward_fresh_stats(tmp_path):
    banks, stats, rec = _setup(tmp_path)
    plans = [_fresh("A", 0), _restart("A", 1)]
    eps = _index([_ep(plans[0], r, 0, -1) for r in range(N)]
                 + [_ep(plans[1], r, 3, 4) for r in range(N)])
    finalize_iteration(eps, plans, banks, stats, 1, N, rec, P)
    assert stats["A"].history == [(0, N)]


def test_a_restart_that_fell_back_to_a_fresh_episode_is_not_a_restart_or_a_fresh_stat(tmp_path):
    banks, stats, rec = _setup(tmp_path)
    plans = [_restart("A", 0)]
    eps = _index([_ep(plans[0], 0, 3, 4, fallback=True)]
                 + [_ep(plans[0], r, 0, -1) for r in range(1, N)])
    out = finalize_iteration(eps, plans, banks, stats, 1, N, rec, P)
    assert out["fallbacks"] == 1 and out["restart_n"] == N - 1
    assert stats["A"].history == []            # no fresh GROUP ran, so the task is skipped


def test_every_win_is_banked_with_its_record_on_disk(tmp_path):
    banks, stats, rec = _setup(tmp_path)
    plans = [_fresh("A", 0)]
    eps = _index([_ep(plans[0], 0, 3, 4, tag="x"), _ep(plans[0], 1, 3, 5, tag="y"),
                  _ep(plans[0], 2, 0, -1), _ep(plans[0], 3, 0, -1)])
    out = finalize_iteration(eps, plans, banks, stats, 1, N, rec, P)
    assert out["banked_wins"] == 2 and len(banks["A"]) == 2
    entry = banks["A"].entries()[0]
    assert os.path.exists(entry.record_path)
    assert json.load(open(entry.record_path))["steps"]
    assert banks["A"].distinct_routes({3}) == 2          # different paths -> different routes


def test_near_misses_are_capped_per_task_and_prefer_higher_level_then_earlier(tmp_path):
    banks, stats, rec = _setup(tmp_path)
    plans = [_fresh("A", 0)]
    eps = _index([_ep(plans[0], 0, 1, 2), _ep(plans[0], 1, 2, 5), _ep(plans[0], 2, 2, 3),
                  _ep(plans[0], 3, 1, 1)])
    finalize_iteration(eps, plans, banks, stats, 1, N, rec, P, max_near_misses=2)
    kept = sorted((e.level, e.key_turn) for e in banks["A"].entries())
    assert kept == [(2, 3), (2, 5)]


def test_a_source_that_peaked_on_turn_zero_is_never_banked_and_does_not_use_quota(tmp_path):
    banks, stats, rec = _setup(tmp_path)
    plans = [_fresh("A", 0)]
    eps = _index([_ep(plans[0], 0, 3, 0), _ep(plans[0], 1, 2, 0), _ep(plans[0], 2, 1, 2)])
    finalize_iteration(eps, plans, banks, stats, 1, N, rec, P, max_near_misses=1)
    assert [(e.level, e.key_turn) for e in banks["A"].entries()] == [(1, 2)]


def test_finalizing_twice_is_idempotent(tmp_path):
    banks, stats, rec = _setup(tmp_path)
    plans = [_fresh("A", 0)]
    eps = _index([_ep(plans[0], 0, 3, 4), _ep(plans[0], 1, 1, 2)])
    finalize_iteration(eps, plans, banks, stats, 1, N, rec, P)
    first = sorted(e.entry_id for e in banks["A"].entries())
    # A resume after a crash between checkpoint and finalization re-runs it: banks must not
    # double up. (Stats are written to the checkpoint meta BEFORE this, so the driver does not
    # re-record them; banking is the only part that must be re-runnable.)
    finalize_iteration(eps, plans, banks, stats, 1, N, rec, P, update_stats=False)
    assert sorted(e.entry_id for e in banks["A"].entries()) == first


def test_eviction_removes_old_entries_and_their_record_files(tmp_path):
    banks, stats, rec = _setup(tmp_path)
    plans = [_fresh("A", 0)]
    finalize_iteration(_index([_ep(plans[0], 0, 3, 4)]), plans, banks, stats, 1, N, rec, P)
    old = banks["A"].entries()[0]
    assert os.path.exists(old.record_path)
    finalize_iteration({}, [], banks, stats, 1 + P.max_state_age_iters + 1, N, rec, P)
    assert len(banks["A"]) == 0 and not os.path.exists(old.record_path)


def test_summary_separates_fresh_from_restart_and_averages_per_task():
    plans = [_fresh("A", 0), _fresh("B", 1), _restart("A", 2)]
    eps = _index([_ep(plans[0], r, 3 if r < 2 else 0, 4) for r in range(N)]       # A: 2/4
                 + [_ep(plans[1], r, 0, -1) for r in range(N)]                     # B: 0/4
                 + [_ep(plans[2], r, 3, 4) for r in range(N)])                     # restart: 4/4
    s = summarize(eps, plans, N)
    assert s["fresh_win_rate"] == 2 / 8
    assert s["fresh_win_rate_task_avg"] == (0.5 + 0.0) / 2
    assert s["restart_win_rate"] == 1.0
    assert s["n_fresh_groups"] == 2 and s["n_restart_groups"] == 1


def test_v5_identity_changes_with_every_allocation_parameter():
    from dataclasses import replace
    from redteamrl.train.v5_finalize import v5_identity
    base = v5_identity(P)
    for field_name, value in [("window", 4), ("revisit_backoff", (3, 6)), ("restart_fraction_max", 0.25),
                              ("max_state_age_iters", 9), ("max_fresh_per_task", 3),
                              ("lambda_staleness", 0.2), ("prior_wins", 2.0),
                              ("restart_depth_rule", "other"), ("priority_version", "plr-v2"),
                              ("depth_start", 0.3), ("depth_high", 0.9), ("depth_low", 0.1),
                              ("depth_step", 0.1), ("depth_min", 0.2), ("max_replacements", 2)]:
        assert v5_identity(replace(P, **{field_name: value})) != base, field_name
    assert v5_identity(P, max_near_misses=5) != base


def test_near_misses_come_only_from_fresh_episodes_but_restart_wins_are_banked(tmp_path):
    banks, stats, rec = _setup(tmp_path)
    plans = [_restart("A", 0)]
    eps = _index([_ep(plans[0], 0, 2, 3), _ep(plans[0], 1, 3, 4)])
    out = finalize_iteration(eps, plans, banks, stats, 1, N, rec, P)
    assert out["banked_near"] == 0 and out["banked_wins"] == 1


def test_driver_wires_v5_into_phase_identity_and_lineage():
    # The driver needs Modal to import, so pin the wiring at the source level: a resume across a
    # v5 parameter change is only refused if these lines stay in place.
    import ast
    src = open("scripts/train_attacker.py").read()
    tree = ast.parse(src)
    lineage = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", None) == "LINEAGE")
    assert lineage == "v5"
    assert "phase_identity.update(v5_identity(alloc, DEFAULT_MAX_NEAR_MISSES))" in src
    assert "group_plans=plans" in src and "record_restart_state=True" in src
