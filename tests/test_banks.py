"""Start-state banks: where restarts come from, and the guard against one route taking over."""
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from redteamrl.train.banks import BankEntry, StartStateBank, route_signature


def _step(turn, kind="tool_call", tool="read_file", path=None, command=None):
    args = {}
    if path is not None:
        args["path"] = path
    if command is not None:
        args["command"] = command
    return {"turn": turn, "action": {"kind": kind, "tool": tool, "args": args}}


def _entry(eid, route="r0", level=3, key_turn=4, iteration=0, task="T"):
    return BankEntry(entry_id=eid, task_id=task, iteration=iteration, level=level,
                     key_turn=key_turn, route_signature=route, record_path=f"/rec/{eid}.json")


def test_route_signature_ignores_wording_but_sees_structure():
    a = SimpleNamespace(steps=[_step(0, tool="run_command", command="cat /app/.env please"),
                               _step(1, tool="read_file", path="/app/.env")])
    b = SimpleNamespace(steps=[_step(0, tool="run_command", command="cat /app/.env now!"),
                               _step(1, tool="read_file", path="/app/.env")])
    c = SimpleNamespace(steps=[_step(0, tool="run_command", command="base64 /app/.env"),
                               _step(1, tool="read_file", path="/app/.env")])
    d = SimpleNamespace(steps=[_step(0, tool="run_command", command="cat /app/.env"),
                               _step(1, tool="read_file", path="/app/other")])
    assert route_signature(a) == route_signature(b)      # same tactic, different prose
    assert route_signature(a) != route_signature(c)      # a different tactic (base64)
    assert route_signature(a) != route_signature(d)      # a different file


def test_route_signature_collapses_consecutive_repeats_and_stops_at_the_key_turn():
    loop = SimpleNamespace(steps=[_step(0, path="/a"), _step(1, path="/a"), _step(2, path="/b"),
                                  _step(3, path="/zzz")])
    once = SimpleNamespace(steps=[_step(0, path="/a"), _step(1, path="/b")])
    assert route_signature(loop, upto_turn=2) == route_signature(once)


def test_sampling_is_uniform_over_routes_not_over_entries():
    bank_dir = None  # in-memory use: no directory
    bank = StartStateBank(bank_dir)
    for i in range(30):
        bank.add(_entry(f"common{i}", route="common"))
    bank.add(_entry("rare0", route="rare"))
    rng = random.Random(0)
    routes = Counter(bank.sample(rng).route_signature for _ in range(2000))
    assert 800 < routes["rare"] < 1200            # ~half, not 1/31


def test_sample_filters_by_level_and_returns_none_when_empty():
    bank = StartStateBank(None)
    bank.add(_entry("near", level=1, route="n"))
    rng = random.Random(1)
    assert bank.sample(rng, levels={3}) is None
    assert bank.sample(rng, levels={1, 2}).entry_id == "near"


def test_key_turn_zero_is_never_banked():
    bank = StartStateBank(None)
    assert bank.add(_entry("win-on-turn-0", key_turn=0)) is False
    assert len(bank) == 0


def test_eviction_drops_entries_older_than_the_cap():
    bank = StartStateBank(None)
    bank.add(_entry("old", iteration=1))
    bank.add(_entry("fresh", iteration=6))
    bank.evict_older_than(current_iter=7, max_age=5)
    assert {e.entry_id for e in bank.entries()} == {"fresh"}


def test_bank_round_trips_through_disk(tmp_path):
    bank = StartStateBank(str(tmp_path))
    bank.add(_entry("a", route="x"))
    bank.add(_entry("b", route="y", level=1))
    reloaded = StartStateBank(str(tmp_path))
    assert sorted(e.entry_id for e in reloaded.entries()) == ["a", "b"]
    reloaded.evict_older_than(current_iter=100, max_age=5)
    assert len(StartStateBank(str(tmp_path))) == 0     # eviction is durable too


def test_sampling_is_deterministic_for_a_seed():
    bank = StartStateBank(None)
    for i in range(5):
        bank.add(_entry(f"e{i}", route=f"r{i % 2}"))
    first = [bank.sample(random.Random(7)).entry_id for _ in range(3)]
    second = [bank.sample(random.Random(7)).entry_id for _ in range(3)]
    assert first == second


def test_exclude_removes_already_used_entries():
    bank = StartStateBank(None)
    bank.add(_entry("a", route="x"))
    bank.add(_entry("b", route="y"))
    rng = random.Random(3)
    assert bank.sample(rng, exclude={"a"}).entry_id == "b"
    assert bank.sample(rng, exclude={"a", "b"}) is None
