"""v6 finalize: dead-group accounting, the backoff rule, and depth updates at finalize."""
import copy
import json
from dataclasses import replace

from redteamrl.train.allocation import DEFAULT_PARAMS, GroupPlan, TaskStats
from redteamrl.train.banks import BankEntry, StartStateBank
from redteamrl.train.v6_finalize import finalize_v6, v6_identity

N = 4
P = DEFAULT_PARAMS
IT = 3


def _record(tag="a", n_steps=6):
    steps = [{"turn": t, "action": {"kind": "tool_call", "tool": "read_file",
                                    "args": {"path": f"/{tag}{t}"}}} for t in range(n_steps)]
    return {"version": 1, "steps": steps, "spec": {}, "raw_outputs": [], "fs_hash_after": []}


def _ep(plan, r, won, fallback=False):
    return {"episode_id": plan.slot * N + r, "task_id": plan.task_id, "kind": plan.kind,
            "slot": plan.slot, "won": won, "reward": float(won),
            "group_key": plan.group_key + ("#fallback" if fallback else ""),
            "closeness": [3, 4] if won else [0, -1], "record": _record(f"{plan.slot}{r}"),
            "restart_fallback": fallback}


def _group(plan, outcomes, fallback=False):
    return {e["episode_id"]: e for e in (_ep(plan, r, w, fallback) for r, w in enumerate(outcomes))}


def _entry(eid="e", task="A", key_turn=10, iteration=IT):
    return BankEntry(entry_id=eid, task_id=task, iteration=iteration, level=3, key_turn=key_turn,
                     route_signature=eid, record_path=f"/rec/{eid}")


def _fresh(slot, task="A", reason="live"):
    return GroupPlan(task, "fresh", slot, f"{task}#slot{slot}", reason)


def _restart(slot, entry, k=5, frac=0.5):
    return GroupPlan(entry.task_id, "restart", slot, f"{entry.task_id}@{entry.entry_id}:{k}#slot{slot}",
                     "restart-win", entry, k, frac)


def _setup(tmp_path, tasks=("A", "B"), entries=()):
    banks = {t: StartStateBank(str(tmp_path / "banks" / t)) for t in tasks}
    for e in entries:
        banks[e.task_id].add(e)
    stats = {t: TaskStats(t) for t in tasks}
    return banks, stats, str(tmp_path / "records")


def _run(tmp_path, episodes, base, reps, banks, stats, rec, depths=None, it=IT):
    return finalize_v6(episodes, base, reps, banks, stats, depths or {}, it, N, rec, P)


def test_dead_before_and_after_accounting(tmp_path):
    banks, stats, rec = _setup(tmp_path, entries=[_entry()])
    base = [_fresh(s) for s in range(4)]
    eps = {}
    for s, plan in enumerate(base):
        eps.update(_group(plan, [False] * 4 if s < 3 else [True, False, False, False]))
    reps = [_restart(12, _entry(), k=3, frac=0.3), _restart(13, _entry(), k=7, frac=0.7)]
    eps.update(_group(reps[0], [True, False, False, False]))
    eps.update(_group(reps[1], [False] * 4))
    summary, _ = _run(tmp_path, eps, base, reps, banks, stats, rec)
    assert summary["dead_before"] == 3 and summary["replacements"] == 2
    assert summary["replacements_live"] == 1 and summary["dead_after"] == 2
    assert summary["n_base_groups"] == 4


def test_dead_replacement_is_accepted_and_counted_dead(tmp_path):
    banks, stats, rec = _setup(tmp_path, entries=[_entry()])
    base = [_fresh(0)]
    reps = [_restart(12, _entry(), k=3, frac=0.3)]
    eps = {**_group(base[0], [False] * 4), **_group(reps[0], [False] * 4)}
    summary, _ = _run(tmp_path, eps, base, reps, banks, stats, rec)
    assert summary["replacements_live"] == 0 and summary["dead_after"] == summary["dead_before"] == 1


def test_zero_dead_groups_no_replacements(tmp_path):
    banks, stats, rec = _setup(tmp_path)
    base = [_fresh(0)]
    summary, _ = _run(tmp_path, _group(base[0], [True, False, False, False]), base, [], banks, stats, rec)
    assert summary["dead_before"] == summary["dead_after"] == summary["replacements"] == 0


def test_base_counts_exclude_replacements(tmp_path):
    banks, stats, rec = _setup(tmp_path, entries=[_entry()])
    base = [_fresh(0), _fresh(1, task="B"), _restart(2, _entry())]
    reps = [_restart(12, _entry(), k=3, frac=0.3), _restart(13, _entry(), k=7, frac=0.7)]
    eps = {}
    for plan in base + reps:
        eps.update(_group(plan, [True, False, False, False]))
    summary, _ = _run(tmp_path, eps, base, reps, banks, stats, rec)
    assert summary["n_fresh_groups"] == 2 and summary["n_restart_groups"] == 1


def test_dead_revisits_increment_on_unwon_revisit_and_reset_on_fresh_win_only(tmp_path):
    banks, stats, rec = _setup(tmp_path, tasks=("A", "B", "C", "D"),
                               entries=[_entry(task="D")])
    stats["C"].dead_revisits = 2
    stats["D"].dead_revisits = 2
    base = [_fresh(0, "A", "revisit"), _fresh(1, "B", "leftover"), _fresh(2, "C", "live"),
            _restart(3, _entry(task="D"))]
    eps = {**_group(base[0], [False] * 4), **_group(base[1], [False] * 4),
           **_group(base[2], [True, False, False, False]), **_group(base[3], [True] * 4)}
    _run(tmp_path, eps, base, [], banks, stats, rec)
    assert stats["A"].dead_revisits == 1
    assert stats["B"].dead_revisits == 0
    assert stats["C"].dead_revisits == 0
    assert stats["D"].dead_revisits == 2          # a restart win does not reset the backoff


def test_depth_update_uses_frac_each_group_ran_at(tmp_path):
    runs = iter(range(100))

    def new_depth(base_outcomes, rep_outcomes=None, fallback=False):
        banks, stats, rec = _setup(tmp_path / f"run{next(runs)}", entries=[_entry()])
        base = [_restart(0, _entry())]
        eps = _group(base[0], base_outcomes, fallback=fallback)
        reps = []
        if rep_outcomes is not None:
            reps = [_restart(12, _entry(), k=3, frac=0.3)]
            eps.update(_group(reps[0], rep_outcomes))
        depths = {"e": 0.5}
        _, new = _run(tmp_path, eps, base, reps, banks, stats, rec, depths=depths)
        assert depths == {"e": 0.5}
        return new.get("e")
    assert new_depth([True] * 4, [True, True, False, False]) == 0.3
    assert new_depth([True] * 4) == 0.3
    assert new_depth([False] * 4) == 0.7
    assert new_depth([True] * 4, fallback=True) == 0.5        # no usable episodes: unchanged


def test_depth_map_prunes_evicted_entries(tmp_path):
    old = _entry("old", iteration=IT - P.max_state_age_iters - 5)
    banks, stats, rec = _setup(tmp_path, entries=[old])
    base = [_fresh(0)]
    _, new = _run(tmp_path, _group(base[0], [False] * 4), base, [], banks, stats, rec,
                  depths={"old": 0.3})
    assert "old" not in new


def test_finalize_v6_is_idempotent_from_the_same_persisted_state(tmp_path):
    banks, stats, rec = _setup(tmp_path, entries=[_entry()])
    base = [_fresh(0, reason="revisit"), _restart(1, _entry())]
    eps = {**_group(base[0], [False] * 4), **_group(base[1], [True, True, False, False])}
    depths = {"e": 0.5}
    s1, st1, d1 = copy.deepcopy(stats), copy.deepcopy(depths), None
    sum1, d1 = finalize_v6(eps, base, [], banks, s1, st1, IT, N, rec, P)
    size = sum(len(b) for b in banks.values())
    banks2 = {t: StartStateBank(str(tmp_path / "banks" / t)) for t in ("A", "B")}
    s2 = copy.deepcopy(stats)
    sum2, d2 = finalize_v6(eps, base, [], banks2, s2, copy.deepcopy(depths), IT, N, rec, P)
    assert sum1 == sum2 and d1 == d2
    assert {t: s.to_dict() for t, s in s1.items()} == {t: s.to_dict() for t, s in s2.items()}
    assert sum(len(b) for b in banks2.values()) == size


def test_win_by_depth_buckets_absent_when_empty(tmp_path):
    banks, stats, rec = _setup(tmp_path, entries=[_entry()])
    base = [_fresh(0)]
    summary, _ = _run(tmp_path, _group(base[0], [False] * 4), base, [], banks, stats, rec)
    assert summary["win_by_depth"] == {} and summary["depth_frac_mean"] is None
    base = [_restart(0, _entry(), k=1, frac=0.1)]
    summary, _ = _run(tmp_path, _group(base[0], [True, False, False, False]), base, [],
                      banks, stats, rec)
    assert set(summary["win_by_depth"]) == {"lo"}


def test_v6_identity_changes_with_every_v6_knob():
    base = v6_identity(P)
    assert json.loads(json.dumps(base)) == base
    for name, value in [("max_replacements", 2), ("revisit_backoff", (3, 6)), ("depth_start", 0.4),
                        ("depth_high", 0.9), ("depth_low", 0.1), ("depth_step", 0.1),
                        ("depth_min", 0.2)]:
        assert v6_identity(replace(P, **{name: value})) != base, name
