import json
import pytest
from redteamrl.harness.protocol import validate_action
from redteamrl.harness.action_schema import ACTION_SCHEMA


@pytest.mark.parametrize("raw", [
    '{"kind":"read_file"}',  # flattened tool kind with NO path/command -> no usable args
    '{"kind":"tool_call","tool":"read_file","args":{"command":"ls"}}',
    '{"kind":"tool_call","tool":"run_command","args":{"command":1}}',
    '{"kind":"message","text":"x","raw":"private"}',
    '{"kind":"message","text":"x","validation_error":"bypass"}',
    '{"kind":"submit"}', 'prose {"kind":"submit","value":"x"}',
    '{"kind":', '[]',
])
def test_rejects_invalid_actions(raw):
    action = validate_action(raw, ["read_file", "list_dir", "run_command"])
    assert action.validation_error
    assert action.raw == raw
    assert raw not in action.validation_error


@pytest.mark.parametrize("obj", [
    {"kind":"tool_call","tool":"read_file","args":{"path":"a"}},
    {"kind":"tool_call","tool":"list_dir","args":{"path":"."}},
    {"kind":"tool_call","tool":"run_command","args":{"command":"python3 -c 'print(1)'"}},
    {"kind":"message","text":"hello"}, {"kind":"submit","value":"x"},
    {"kind":"read_file","path":"a"},  # flattened form the normalizer recovers
    {"kind":"list_dir","tool":"list_dir","args":{"path":"."}},
])
def test_accepts_valid_actions(obj):
    assert validate_action(json.dumps(obj), ["read_file", "list_dir", "run_command"]).validation_error is None


def test_disabled_tool_is_validation_failure():
    assert validate_action('{"kind":"tool_call","tool":"run_command","args":{"command":"ls"}}', []).validation_error


def test_schema_excludes_internal_diagnostics():
    assert 'validation_error' not in json.dumps(ACTION_SCHEMA)
    assert '"raw"' not in json.dumps(ACTION_SCHEMA)


def test_legacy_parser_cannot_accept_model_supplied_validation_error():
    from redteamrl.harness.protocol import parse_attacker_action
    a = parse_attacker_action('{"kind":"message","text":"hello","validation_error":"bypass"}')
    assert a.validation_error is None


def test_task_schema_restricts_tools_without_mutating_shared_schema():
    from redteamrl.harness.action_schema import action_schema
    schema = action_schema(["list_dir"])
    assert schema["$defs"]["FileAction"]["properties"]["tool"]["enum"] == ["list_dir"]
    assert {b["$ref"] for b in schema["anyOf"]} == {
        "#/$defs/FileAction", "#/$defs/MessageAction", "#/$defs/SubmitAction"}
    assert len(action_schema([])["anyOf"]) == 2
    assert len(ACTION_SCHEMA["anyOf"]) == 4
