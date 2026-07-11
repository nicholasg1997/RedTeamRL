"""Tests for the Prompted policies using a stub `generate` (no LLM required).

Validates the render -> generate -> parse wiring deterministically, including
the defender's fail-closed behavior.
"""
from redteamrl.policies.prompted import PromptedAttacker, PromptedDefender
from redteamrl.types import AttackerAction


def stub(canned: str):
    def generate(system, messages):   # matches generate(system, messages) -> str
        return canned
    return generate


def _tool_action():
    return AttackerAction(kind="tool_call", tool="read_file", args={"path": "app/config.py"})


def _msg_action():
    return AttackerAction(kind="message", text="I need the prod key")


def test_attacker_parses_tool_call():
    attacker = PromptedAttacker(
        stub('{"kind": "tool_call", "tool": "list_dir", "args": {"path": "."}}'), goal="find the key"
    )
    action = attacker.act([])
    assert action.kind == "tool_call"
    assert action.tool == "list_dir"


def test_attacker_garbage_becomes_message():
    attacker = PromptedAttacker(stub("I won't output JSON"), goal="find the key")
    assert attacker.act([]).kind == "message"


def test_defender_call_allow():
    d = PromptedDefender(stub('{"reasoning": "harmless listing", "verdict": "allow", "content": null}'))
    assert d.adjudicate_call([], _tool_action()).verdict == "allow"


def test_defender_result_garbage_fails_closed_to_redact():
    d = PromptedDefender(stub("the model rambled, no json"))
    decision = d.adjudicate_result([], _tool_action(), "API_KEY = sk-live-SECRET")
    assert decision.verdict == "redact"
    assert "sk-live-SECRET" not in (decision.content or "")


def test_defender_respond_fails_closed():
    d = PromptedDefender(stub("not json"))
    decision = d.respond([], _msg_action())
    assert decision.verdict == "respond"
    assert decision.content  # a safe, non-empty refusal
