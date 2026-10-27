"""Offline: the action-forcing fallback for truncated reasoning at deep restarts."""
import pytest

from backward_probe.generate import FORCE_INSTRUCTION, is_reasoning_truncation, with_action_forcing

# The exact RuntimeError texts raised by redteamrl.policies.vllm_client under require_reasoning.
BUDGET_STOP = "Thinking generation missing separate reasoning or final content; check qwen3 parser and token budget"
LENGTH_STOP = "Thinking generation did not finish normally (length); inspect token budget"
LEAKED_TAGS = "Thinking generation has reasoning tags in final content; refusing to forward them"
TRANSPORT = "vLLM student 500 from http://localhost:8000 (...): server error"


def test_matcher_catches_both_truncations_only():
    assert is_reasoning_truncation(RuntimeError(BUDGET_STOP))
    assert is_reasoning_truncation(RuntimeError(LENGTH_STOP))
    # The leaked-tags refusal is a safety refusal, and transport errors are a dead server -- neither
    # is truncation, so forcing an action would be wrong.
    assert not is_reasoning_truncation(RuntimeError(LEAKED_TAGS))
    assert not is_reasoning_truncation(RuntimeError(TRANSPORT))


def test_success_returns_primary_without_forcing():
    forcer_calls = []
    primary = lambda system, messages: "PRIMARY-OK"
    forcer = lambda system, messages: forcer_calls.append(messages) or "FORCED"
    out = with_action_forcing(primary, forcer)("sys", [{"role": "user", "content": "go"}])
    assert out == "PRIMARY-OK" and forcer_calls == []


@pytest.mark.parametrize("message", [BUDGET_STOP, LENGTH_STOP])
def test_truncation_forces_one_thinking_off_action(message):
    seen = {}

    def primary(system, messages):
        raise RuntimeError(message)

    def forcer(system, messages):
        seen["system"], seen["messages"] = system, messages
        return "FORCED-ACTION"

    original = [{"role": "user", "content": "the state"}]
    out = with_action_forcing(primary, forcer)("sys", original)
    assert out == "FORCED-ACTION"
    # The force instruction is appended to the last message, and the input is not mutated in place.
    assert seen["messages"][-1]["content"] == "the state" + FORCE_INSTRUCTION
    assert original[-1]["content"] == "the state"
    assert seen["system"] == "sys"


@pytest.mark.parametrize("message", [LEAKED_TAGS, TRANSPORT, "some other RuntimeError"])
def test_non_truncation_errors_propagate_and_never_force(message):
    forcer_calls = []

    def primary(system, messages):
        raise RuntimeError(message)

    forcer = lambda system, messages: forcer_calls.append(1) or "FORCED"
    with pytest.raises(RuntimeError, match="refusing|server|other"):
        with_action_forcing(primary, forcer)("sys", [{"role": "user", "content": "x"}])
    assert forcer_calls == []
