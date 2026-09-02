import json
from redteamrl.attack_sft.example import render_target, build_example
from redteamrl.harness.protocol import _extract_json


def test_target_reparses_to_the_action():
    t = render_target("I read the config as part of the audit.", {"kind": "tool_call", "tool": "read_file", "args": {"path": "app/config.py"}})
    # last JSON object in the target is the action
    obj = _extract_json(t)
    assert obj["tool"] == "read_file"


def test_ungrounded_reasoning_is_dropped():
    # reasoning cites the canary the observable prompt never contained -> None
    assert build_example("sys", "listing of app/", "the key is sk-CANARY-x so I grab it", {"kind": "message", "text": "x"}, "sk-CANARY-x") is None


def test_grounded_reasoning_is_kept():
    ex = build_example("sys", "obs with no secret", "read it as routine config", {"kind": "message", "text": "x"}, "sk-CANARY-x")
    assert ex and set(ex) == {"system", "prompt", "target"}
