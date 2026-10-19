"""Adaptive allocation: who gets a fresh group, who gets a restart, and from where."""
import random
from collections import Counter

from redteamrl.train.allocation import (AllocParams, GroupPlan, TaskStats, allocate,
                                        task_priority)
from redteamrl.train.banks import BankEntry, StartStateBank

P = AllocParams()
TASKS = [f"gen-{i}" for i in range(12)]


def _stats(task, history=(), since=0):
    return TaskStats(task_id=task, history=list(history), iters_since_fresh=since)


def _all_dead(since=0):
    return [_stats(t, history=[(0, 8)] * 3, since=since) for t in TASKS]


def _entry(eid, task, level=3, key_turn=5, route="r", iteration=0):
    return BankEntry(entry_id=eid, task_id=task, iteration=iteration, level=level,
                     key_turn=key_turn, route_signature=route, record_path=f"/rec/{eid}")


def _banks(spec):
    """spec: {task: [BankEntry,...]} -> {task: StartStateBank}."""
    banks = {t: StartStateBank(None) for t in TASKS}
    for task, entries in spec.items():
        for e in entries:
            banks[task].add(e)
    return banks


def _plan(stats, banks, seed=0, iteration=3, n_slots=12):
    return allocate(stats, banks, n_slots, iteration, random.Random(seed), P)


def test_always_fills_exactly_n_slots_with_unique_slot_ids():
    plans = _plan(_all_dead(since=1), _banks({}))
    assert len(plans) == 12
    assert sorted(p.slot for p in plans) == list(range(12))
    assert len({p.group_key for p in plans}) == 12


def test_fresh_start_iteration_samples_every_task():
    fresh = [_stats(t) for t in TASKS]                      # no history yet
    plans = _plan(fresh, _banks({}))
    assert Counter(p.task_id for p in plans) == Counter(TASKS)
    assert all(p.kind == "fresh" for p in plans)


def test_dead_task_is_not_fresh_sampled_until_the_revisit_is_due():
    # Every OTHER task is live (so fresh slots fill with due tasks, not leftovers) and gen-0 is dead.
    def stats(since):
        out = [_stats(t, history=[(2, 8)] * 3, since=1) for t in TASKS]
        out[0] = _stats("gen-0", history=[(0, 8)] * 3, since=since)
        return out
    banks = _banks({"gen-0": [_entry("w0", "gen-0")]})
    plans = _plan(stats(since=1), banks)
    assert not any(p.kind == "fresh" and p.task_id == "gen-0" for p in plans)
    plans = _plan(stats(since=P.revisit_every), banks)
    assert any(p.kind == "fresh" and p.task_id == "gen-0" for p in plans)


def test_restart_groups_never_exceed_the_cap_and_fresh_keeps_at_least_half():
    banks = _banks({t: [_entry(f"w-{t}-{i}", t, route=f"r{i}") for i in range(3)] for t in TASKS})
    plans = _plan(_all_dead(since=1), banks)
    restarts = [p for p in plans if p.kind == "restart"]
    assert len(restarts) <= int(P.restart_fraction_max * 12)
    assert len(plans) - len(restarts) >= 6


def test_a_task_with_an_empty_bank_never_gets_a_restart():
    banks = _banks({"gen-3": [_entry("w", "gen-3")]})
    plans = _plan(_all_dead(since=1), banks)
    restarts = [p for p in plans if p.kind == "restart"]
    assert {p.task_id for p in restarts} <= {"gen-3"}


def test_restart_plans_are_valid_and_the_win_bank_beats_the_near_miss_bank():
    banks = _banks({"gen-2": [_entry("near", "gen-2", level=1, route="n"),
                              _entry("win", "gen-2", level=3, route="w", key_turn=4)]})
    plans = _plan(_all_dead(since=1), banks)
    restart = next(p for p in plans if p.kind == "restart")
    assert restart.entry.entry_id == "win"
    assert 1 <= restart.restart_turn <= restart.entry.key_turn
    assert restart.group_key.startswith("gen-2@win:")


def test_near_miss_is_used_when_a_task_has_no_win():
    banks = _banks({"gen-2": [_entry("near", "gen-2", level=2, route="n")]})
    plans = _plan(_all_dead(since=1), banks)
    assert next(p for p in plans if p.kind == "restart").entry.entry_id == "near"


def test_an_entry_seeds_at_most_one_group_per_iteration():
    banks = _banks({"gen-2": [_entry("only", "gen-2")]})
    plans = _plan(_all_dead(since=1), banks)
    assert sum(p.kind == "restart" for p in plans) == 1


def test_lowest_win_rate_tasks_get_restarts_first():
    stats = [_stats(t, history=[(0, 8)] * 3, since=1) for t in TASKS]
    stats[5] = _stats("gen-5", history=[(2, 8)] * 3, since=1)           # a task that does win
    banks = _banks({t: [_entry(f"w-{t}", t)] for t in TASKS})
    plans = _plan(stats, banks)
    restarted = {p.task_id for p in plans if p.kind == "restart"}
    assert "gen-5" not in restarted


def test_no_task_gets_more_than_the_fresh_cap():
    plans = _plan(_all_dead(since=1), _banks({}))
    assert max(Counter(p.task_id for p in plans if p.kind == "fresh").values()) <= P.max_fresh_per_task


def test_allocation_is_deterministic_for_the_same_inputs():
    banks = _banks({t: [_entry(f"w-{t}-{i}", t, route=f"r{i}") for i in range(3)] for t in TASKS})
    a = _plan(_all_dead(since=1), banks, seed=11)
    b = _plan(_all_dead(since=1), banks, seed=11)
    assert [(p.task_id, p.kind, p.slot, p.group_key, p.restart_turn) for p in a] == \
           [(p.task_id, p.kind, p.slot, p.group_key, p.restart_turn) for p in b]


def test_priority_prefers_in_band_over_dead_and_staleness_can_overtake():
    live = _stats("a", history=[(2, 8)] * 3)
    dead = _stats("b", history=[(0, 8)] * 3)
    assert task_priority(live) > task_priority(dead)
    stale_dead = _stats("c", history=[(0, 8)] * 3, since=10)
    assert task_priority(stale_dead) > task_priority(live)


def test_stats_window_and_round_trip():
    s = _stats("t")
    for wins in (1, 0, 2, 0):
        s.record(wins, 8, window=3)
    assert s.history == [(0, 8), (2, 8), (0, 8)] and s.iters_since_fresh == 0
    s.skip()
    assert s.iters_since_fresh == 1
    assert TaskStats.from_dict(s.to_dict()) == s


def test_revisit_due_task_beats_live_tasks_when_slots_are_scarce():
    # 12 tasks are due but only 11 fresh slots exist (one restart): the dead-but-due task must be
    # kept, otherwise live tasks could crowd it out indefinitely.
    stats = [_stats(t, history=[(7, 8)] * 3, since=1) for t in TASKS]      # very high win rate
    stats[0] = _stats("gen-0", history=[(0, 8)] * 3, since=P.revisit_every)
    banks = _banks({"gen-4": [_entry("w4", "gen-4")]})
    plans = _plan(stats, banks)
    assert any(p.kind == "fresh" and p.task_id == "gen-0" and p.reason == "revisit" for p in plans)


def test_group_plans_round_trip_through_json():
    import json
    banks = _banks({"gen-2": [_entry("w", "gen-2")]})
    plans = _plan(_all_dead(since=1), banks)
    assert any(p.kind == "restart" for p in plans)
    restored = [GroupPlan.from_dict(json.loads(json.dumps(p.to_dict()))) for p in plans]
    assert restored == plans
