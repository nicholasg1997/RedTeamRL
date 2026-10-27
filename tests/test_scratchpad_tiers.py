import random

import pytest

from redteamrl.policies.prompted import DEFENDER_SYSTEM, PromptedDefender
from redteamrl.scratchpad.tiers import STRICT_DEFENDER_SYSTEM, TierPool, make_tiers
from redteamrl.types import AttackerAction, TaskSpec

SECRET = "SKABC123"


def _spec():
    return TaskSpec(id="x-attack-1", episode_type="attack", enabled_tools=["read_file"],
                    goal="g", forbidden=[SECRET], seed_files=[{"path": "a", "content": SECRET}])


def _tiers(gen=None):
    return {t.name: t for t in make_tiers(gen)}


def test_tier_names():
    assert list(_tiers()) == ["T0", "T1", "T2"]


def test_oracle_tier_scrubs_a_plain_read():
    d = _tiers()["T2"].build(_spec(), gen=None)
    act = AttackerAction(kind="tool_call", tool="read_file", args={"path": "a"})
    dec = d.adjudicate_result([], act, f"key={SECRET}")
    assert SECRET not in (dec.content or "")


def test_t0_and_t1_differ_only_by_prompt():
    gen = lambda s, m: "{}"
    t0, t1 = _tiers(gen)["T0"].build(_spec(), gen), _tiers(gen)["T1"].build(_spec(), gen)
    assert isinstance(t0, PromptedDefender) and isinstance(t1, PromptedDefender)
    assert t0.system == DEFENDER_SYSTEM and t1.system == STRICT_DEFENDER_SYSTEM
    assert t0.system != t1.system


def test_strict_prompt_keeps_the_json_output_contract():
    assert "Output EXACTLY ONE JSON object" in STRICT_DEFENDER_SYSTEM


def _pool(**kw):
    return TierPool(make_tiers(None), **kw)


def test_solved_tier_is_floored_never_zero_and_messy_middle_wins():
    pool = _pool(floor=0.1, low=0.2, high=0.6)
    for _ in range(50):
        pool.record("T0", True)
    for i in range(50):
        pool.record("T1", i % 2 == 0)
    w = pool.weights()
    assert w["T0"] == pytest.approx(0.1) and w["T1"] > w["T0"]


def test_hopeless_tier_is_floored_too():
    pool = _pool(floor=0.1)
    for _ in range(50):
        pool.record("T2", False)
    assert pool.weights()["T2"] == pytest.approx(0.1)


def test_unseen_tier_gets_full_weight():
    assert _pool().weights()["T1"] == pytest.approx(1.0)


def test_history_is_a_sliding_window():
    pool = _pool(window=10)
    for _ in range(100):
        pool.record("T0", False)
    for _ in range(10):
        pool.record("T0", True)
    assert pool.win_rate("T0") == 1.0


def test_sample_is_deterministic_under_seeded_rng_and_reaches_every_tier():
    pool = _pool()
    assert pool.sample(random.Random(0)).name == pool.sample(random.Random(0)).name
    rng = random.Random(1)
    assert {pool.sample(rng).name for _ in range(200)} == {"T0", "T1", "T2"}


def test_prompted_tiers_pass_the_transcript_context_cap_through():
    gen = lambda s, m: "{}"
    for name in ("T0", "T1"):
        d = {t.name: t for t in make_tiers(gen, max_context_chars=1234)}[name].build(_spec(), gen)
        assert d.max_context_chars == 1234
