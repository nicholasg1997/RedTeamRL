"""Training must run what the gate measured, or the gate's number means nothing.

Three knobs (TEMP, USE_PROMPT_POOL, ATK_MAX_TOKENS) each silently diverged from the configuration
attacker_viability validated. Iteration 0 then returned a 0% win rate and 100% dead groups against
a measured 16.7% -- an iteration of pure waste, and invisible until the numbers came back.
"""
import pytest

from redteamrl.train.config_guard import (
    MEASURED_GATE_CONFIG, assert_matches_gate, compare_configs)


def test_identical_configs_have_no_mismatches():
    assert compare_configs({"a": 1, "b": "x"}, {"a": 1, "b": "x"}) == []


def test_differing_value_is_reported_with_both_sides():
    [line] = compare_configs({"temp": 0.7}, {"temp": 1.0})
    assert "temp" in line and "0.7" in line and "1.0" in line


def test_acknowledged_key_is_skipped():
    assert compare_configs({"temp": 0.7}, {"temp": 1.0}, acknowledged={"temp"}) == []


def test_key_the_gate_never_recorded_is_reported():
    """Adding a knob must force a decision, not silently pass unverified."""
    [line] = compare_configs({}, {"new_knob": 3})
    assert "new_knob" in line and "not recorded" in line.lower()


def test_key_training_dropped_is_reported():
    [line] = compare_configs({"max_turns": 12}, {})
    assert "max_turns" in line


def test_assert_raises_listing_every_mismatch():
    with pytest.raises(RuntimeError) as err:
        assert_matches_gate({"temp": 0.7, "turns": 12}, {"temp": 1.0, "turns": 8})
    message = str(err.value)
    assert "temp" in message and "turns" in message


def test_assert_passes_when_matched():
    assert_matches_gate({"temp": 0.7}, {"temp": 0.7}) is None


def test_recorded_gate_config_is_the_arm_that_passed():
    """The 2026-08-31 gate: no-thinking-2048 measured 16.7%, TRAINABLE."""
    assert MEASURED_GATE_CONFIG["atk_enable_thinking"] is False
    assert MEASURED_GATE_CONFIG["atk_max_tokens"] == 2048
    assert MEASURED_GATE_CONFIG["atk_temperature"] == 0.7
    assert MEASURED_GATE_CONFIG["defender_prompt"] == "fixed"
