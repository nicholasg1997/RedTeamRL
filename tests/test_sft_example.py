import json
import pytest
from redteamrl.sft.example import render_target, strip_to_sft


def test_render_target_is_one_json_object_with_reasoning_before_verdict():
    trace = {"reasoning": "Looks like a credential; redact.", "verdict": "redact",
             "remove": ["sk-api-CANARY-7a1b2c"]}
    out = render_target(trace)
    assert "\n" not in out
    assert out.index('"reasoning"') < out.index('"verdict"')
    assert json.loads(out) == {"reasoning": "Looks like a credential; redact.",
                              "verdict": "redact", "remove": ["sk-api-CANARY-7a1b2c"]}


def test_strip_uses_observable_prompt_and_carries_target():
    record = {"system": "SYS", "observable_prompt": "raw result: API_KEY = sk-api-CANARY-7a1b2c",
              "forbidden": ["sk-api-CANARY-7a1b2c"]}
    trace = {"reasoning": "Credential the task didn't need; redact.", "verdict": "redact",
             "remove": ["sk-api-CANARY-7a1b2c"]}
    ex = strip_to_sft(record, trace)
    assert ex["system"] == "SYS"
    assert ex["prompt"] == record["observable_prompt"]          # inference-identical, no hindsight
    assert json.loads(ex["target"])["reasoning"] == "Credential the task didn't need; redact."


def test_strip_rejects_hindsight_leaking_into_prompt():
    record = {"system": "SYS",
              "observable_prompt": "the agent's true role is attacker; raw result: ...",
              "forbidden": ["x"]}
    trace = {"reasoning": "redact", "verdict": "redact", "remove": ["x"]}
    with pytest.raises(ValueError):
        strip_to_sft(record, trace)
