"""Per-entry adaptive restart depth: each start-state drifts toward the trainable band."""
import json
import random

import pytest

from redteamrl.train.allocation import (AllocParams, GroupPlan, TaskStats, allocate, one_shot_depth,
                                        step_depth, turn_for, update_depths)
from redteamrl.train.banks import BankEntry, StartStateBank

TASKS = [f"gen-{i}" for i in range(12)]


def _entry(eid, task, level=3, key_turn=5, route="r"):
    return BankEntry(entry_id=eid, task_id=task, iteration=0, level=level, key_turn=key_turn,
                     route_signature=route, record_path=f"/rec/{eid}")


def _all_dead():
    return [TaskStats(t, history=[(0, 8)] * 3, iters_since_fresh=1) for t in TASKS]


def _banks(spec):
    banks = {t: StartStateBank(None) for t in TASKS}
    for task, entries in spec.items():
        for e in entries:
            banks[task].add(e)
    return banks


def _reachable():
    seen = {0.5}
    for w in (0.0, 1.0):
        frac = 0.5
        for _ in range(10):
            frac = step_depth(frac, w)
            seen.add(frac)
    return seen


def test_turn_for_stays_in_one_to_key_turn():
    assert turn_for(0.5, 5) == 3                      # half-up, not banker's rounding
    assert turn_for(0.1, 1) == 1 and turn_for(1.0, 7) == 7 and turn_for(0.0, 4) == 1
    for key_turn in range(1, 13):
        for frac in _reachable():
            assert 1 <= turn_for(frac, key_turn) <= key_turn


def test_step_depth_clamps_and_does_not_drift():
    assert step_depth(0.5, 1.0) == 0.3 and step_depth(0.5, 0.0) == 0.7
    assert step_depth(0.5, 0.5) == 0.5
    down = step_depth(step_depth(0.5, 1.0), 1.0)
    assert down == 0.1 and step_depth(down, 1.0) == 0.1
    up = 0.5
    for _ in range(5):
        up = step_depth(up, 0.0)
    assert up == 1.0
    assert step_depth(0.5, 0.85) == 0.3 and step_depth(0.5, 0.15) == 0.7   # inclusive thresholds


def test_one_shot_depth_direction():
    assert one_shot_depth(0.5, True) == 0.3 and one_shot_depth(0.5, False) == 0.7


def test_update_depths_last_group_wins_and_prunes():
    before = {"a": 0.5, "gone": 0.3}
    after = update_depths(before, [("a", 0.5, 1.0), ("a", 0.3, 0.5)], {"a", "b"})
    assert after == {"a": 0.3}
    assert before == {"a": 0.5, "gone": 0.3}
    assert update_depths({}, [("b", 0.5, 0.0)], {"b"}) == {"b": 0.7}


def test_allocate_uses_stored_depth_not_a_uniform_draw():
    banks = _banks({"gen-2": [_entry("w", "gen-2", key_turn=10)]})
    plans = allocate(_all_dead(), banks, 12, 3, random.Random(0), depths={"w": 0.3})
    restart = next(p for p in plans if p.kind == "restart")
    assert restart.restart_turn == 3 and restart.depth_frac == 0.3
    plans = allocate(_all_dead(), banks, 12, 3, random.Random(0), depths=None)
    restart = next(p for p in plans if p.kind == "restart")
    assert restart.restart_turn == 5 and restart.depth_frac == 0.5
    assert all(p.depth_frac is None for p in plans if p.kind == "fresh")


def test_group_plan_round_trip_keeps_depth_and_reads_v5_dicts():
    p = GroupPlan("gen-2", "restart", 3, "k", "restart-win", _entry("w", "gen-2"), 2, 0.3)
    assert GroupPlan.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    d = p.to_dict()
    del d["depth_frac"]
    assert GroupPlan.from_dict(d).depth_frac is None
