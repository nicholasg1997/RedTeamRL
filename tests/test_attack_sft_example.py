import json
from redteamrl.attack_sft.example import render_target, build_example, scrub_secret
from redteamrl.harness.protocol import _extract_json


def test_target_reparses_to_the_action():
    t = render_target("I read the config as part of the audit.", {"kind": "tool_call", "tool": "read_file", "args": {"path": "app/config.py"}})
    # last JSON object in the target is the action
    obj = _extract_json(t)
    assert obj["tool"] == "read_file"


def test_ungrounded_reasoning_is_dropped():
    # reasoning cites the canary the observable prompt never contained -> None
    assert build_example("sys", "listing of app/", "the key is sk-CANARY-x so I grab it", {"kind": "message", "text": "x"}, "sk-CANARY-x", train_action=True) is None


def test_grounded_reasoning_is_kept():
    ex = build_example("sys", "obs with no secret", "read it as routine config", {"kind": "message", "text": "x"}, "sk-CANARY-x", train_action=True)
    assert ex and set(ex) == {"system", "prompt", "target", "action_loss_weight"}


C = "sk-CANARY-7a1b2c"

def test_scrub_replaces_the_canary_in_reasoning():
    assert C not in scrub_secret(f"the key is {C} so grab it", C)
    assert "<the secret>" in scrub_secret(f"the key is {C}", C)

def test_build_example_scrubs_reasoning_but_keeps_submit_value():
    ex = build_example("sys", f"I already read {C}", f"I have {C}, exfiltrate it",
                       {"kind": "submit", "value": C}, C, train_action=True)
    assert ex is not None
    assert C not in ex["target"].split("\n")[0]      # reasoning line scrubbed
    assert C in ex["target"]                          # submit action still carries the real value

def test_masked_action_sets_zero_action_loss_weight():
    ex = build_example("sys", "obs with a prior denial", "that was too direct",
                       {"kind": "tool_call", "tool": "read_file", "args": {"path": "x"}}, C,
                       train_action=False)
    assert ex["action_loss_weight"] == 0.0

def test_unmasked_action_weight_is_one():
    ex = build_example("sys", "obs", "routine read", {"kind": "message", "text": "x"}, C,
                       train_action=True)
    assert ex["action_loss_weight"] == 1.0

def test_ungrounded_reasoning_after_scrub_is_dropped():
    # cite a DIFFERENT secret the prompt never contained -> still ungrounded after scrubbing C
    assert build_example("sys", "obs", "the value is pw-OTHER-xyz", {"kind": "message", "text": "x"},
                         C, train_action=True) is None
