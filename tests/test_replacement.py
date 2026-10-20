"""Dynamic sampling: replace dead base groups with a bounded, deterministic set of extra groups."""
import json
import random

import pytest

from redteamrl.train.allocation import AllocParams, GroupPlan
from redteamrl.train.banks import BankEntry, StartStateBank
from redteamrl.train.replacement import is_dead, load_or_plan, plan_replacements

N = 4
N_SLOTS = 12
P = AllocParams()


def _entry(eid, task="t", level=3, key_turn=10):
    return BankEntry(entry_id=eid, task_id=task, iteration=0, level=level, key_turn=key_turn,
                     route_signature=eid, record_path=f"/rec/{eid}")


def _fresh(slot, task="t"):
    return GroupPlan(task, "fresh", slot, f"{task}#slot{slot}", "live")


def _restart(slot, entry, k=5, frac=0.5):
    return GroupPlan(entry.task_id, "restart", slot, f"{entry.task_id}@{entry.entry_id}:{k}#slot{slot}",
                     "restart-win", entry, k, frac)


def _ep(plan, r, reward, won, fallback=False):
    return {"episode_id": plan.slot * N + r, "slot": plan.slot, "task_id": plan.task_id,
            "group_key": plan.group_key + ("#fallback" if fallback else ""), "reward": reward,
            "won": won, "restart_fallback": fallback}


def _group(plan, outcomes):
    return {e["episode_id"]: e for e in
            (_ep(plan, r, 1.0 if w else -0.1, w) for r, w in enumerate(outcomes))}


def _banks(entries):
    banks = {}
    for e in entries:
        banks.setdefault(e.task_id, StartStateBank(None)).add(e)
    return banks


def _plan(episodes, plans, banks, depths=None, params=P, seed=0):
    return plan_replacements(episodes, plans, banks, depths or {}, N_SLOTS, N,
                             random.Random(seed), params)


def test_is_dead_uniform_reward_per_bucket():
    p = _fresh(0)
    assert is_dead([_ep(p, r, 0.0, False) for r in range(4)])
    assert not is_dead([_ep(p, 0, 1.0, True)] + [_ep(p, r, 0.0, False) for r in range(1, 4)])
    mixed = [_ep(p, r, -0.1, False) for r in range(3)] + [_ep(p, 3, 1.0, True, fallback=True)]
    assert is_dead(mixed)


def test_dead_restart_all_win_retries_same_entry_earlier():
    e = _entry("w")
    base = _restart(0, e)
    rep = _plan(_group(base, [True] * 4), [base], _banks([e]))
    assert len(rep) == 1 and rep[0].entry.entry_id == "w"
    assert rep[0].depth_frac == 0.3 and rep[0].restart_turn == 3
    assert rep[0].slot == N_SLOTS and rep[0].reason.endswith("-earlier")
    rep = _plan(_group(base, [False] * 4), [base], _banks([e]))
    assert rep[0].depth_frac == 0.7 and rep[0].restart_turn == 7 and rep[0].reason.endswith("-later")


def test_dead_restart_at_clamp_is_conceded():
    e = _entry("w", key_turn=1)
    base = _restart(0, e, k=1)
    assert _plan(_group(base, [True] * 4), [base], _banks([e])) == []


def test_dead_fresh_becomes_restart_from_its_bank_or_is_conceded():
    win = _entry("w")
    base = _fresh(0)
    rep = _plan(_group(base, [False] * 4), [base], _banks([win]))
    assert len(rep) == 1 and rep[0].kind == "restart" and rep[0].entry.entry_id == "w"
    assert rep[0].restart_turn == 5 and rep[0].depth_frac == 0.5
    assert _plan(_group(base, [False] * 4), [base], {}) == []
    assert _plan(_group(base, [True] * 4), [base], _banks([win])) == []
    near = _entry("n", level=1)
    used = _restart(1, win)
    episodes = {**_group(base, [False] * 4), **_group(used, [True, False, False, False])}
    rep = _plan(episodes, [base, used], _banks([win, near]))
    assert [p.entry.entry_id for p in rep] == ["n"]


def test_live_groups_are_never_replaced():
    base = _fresh(0)
    assert _plan(_group(base, [True, False, False, False]), [base], _banks([_entry("w")])) == []


def _six_dead():
    plans = [_fresh(s, task=f"t{s}") for s in range(6)]
    episodes = {}
    for p in plans:
        episodes.update(_group(p, [False] * 4))
    banks = _banks([_entry(f"w{s}", task=f"t{s}") for s in range(6)])
    return episodes, plans, banks


def test_cap_takes_lowest_dead_slots_and_is_deterministic():
    episodes, plans, banks = _six_dead()
    rep = _plan(episodes, plans, banks, seed=7)
    assert [p.task_id for p in rep] == ["t0", "t1", "t2", "t3"]
    assert [p.slot for p in rep] == [N_SLOTS + j for j in range(4)]
    again = _plan(episodes, plans, banks, seed=7)
    assert [p.to_dict() for p in rep] == [p.to_dict() for p in again]


def test_replacement_ids_disjoint_from_base():
    episodes, plans, banks = _six_dead()
    rep = _plan(episodes, plans, banks)
    ids = [p.slot * N + r for p in rep for r in range(N)]
    assert min(ids) >= N_SLOTS * N and len({p.slot for p in rep}) == len(rep)


def test_partial_base_rollout_refuses_to_plan():
    episodes, plans, banks = _six_dead()
    del episodes[0]
    with pytest.raises(KeyError):
        _plan(episodes, plans, banks)


def test_load_or_plan_replays_persisted_plan_without_recomputing(tmp_path):
    episodes, plans, banks = _six_dead()
    path = str(tmp_path / "r" / "replacements.json")
    first = load_or_plan(path, lambda: _plan(episodes, plans, banks))

    def boom():
        raise AssertionError("must not recompute")
    second = load_or_plan(path, boom)
    assert [p.to_dict() for p in first] == [p.to_dict() for p in second]
    assert not list((tmp_path / "r").glob("*.tmp"))


def test_empty_plan_is_persisted_as_empty_list(tmp_path):
    path = str(tmp_path / "replacements.json")
    assert load_or_plan(path, lambda: []) == []
    assert json.load(open(path)) == []
    calls = []
    load_or_plan(path, lambda: calls.append(1) or [])
    assert calls == []
