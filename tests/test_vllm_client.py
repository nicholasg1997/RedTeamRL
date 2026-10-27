import subprocess

import pytest
from unittest.mock import patch, MagicMock
from redteamrl.policies.vllm_client import make_vllm_generate, start_vllm_server, stop_vllm_server

def test_builds_openai_payload_and_extracts_content():
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
    with patch("redteamrl.policies.vllm_client.requests.post", return_value=fake) as post:
        gen = make_vllm_generate("http://localhost:8000", "Qwen/Qwen3-4B", temperature=0.7)
        out = gen("SYS", [{"role": "user", "content": "hi"}])
    assert out == "hello"
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "Qwen/Qwen3-4B"
    assert payload["messages"][0] == {"role": "system", "content": "SYS"}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}
    assert payload["temperature"] == 0.7
    assert post.call_args.args[0] == "http://localhost:8000/v1/chat/completions"

def test_non_200_surfaces_server_body():
    fake = MagicMock()
    fake.status_code = 400
    fake.text = "maximum context length is 8192 tokens, however you requested 9001"
    with patch("redteamrl.policies.vllm_client.requests.post", return_value=fake):
        gen = make_vllm_generate("http://localhost:8000", "Qwen/Qwen3-4B")
        with pytest.raises(RuntimeError, match="maximum context length"):
            gen("SYS", [{"role": "user", "content": "hi"}])


def test_start_vllm_server_returns_when_healthy():
    resp = MagicMock(); resp.status_code = 200
    with patch("redteamrl.policies.vllm_client.subprocess.Popen") as popen, \
         patch("redteamrl.policies.vllm_client.requests.get", return_value=resp):
        popen.return_value.poll.return_value = None
        start_vllm_server("Qwen/Q", 8000, 0.4, max_model_len=4096, max_num_seqs=16)
    cmd = popen.call_args.args[0]
    assert cmd[:3] == ["vllm", "serve", "Qwen/Q"]
    assert "0.4" in cmd and "4096" in cmd and "8000" in cmd
    assert "--max-num-seqs" in cmd and "16" in cmd


def test_start_vllm_server_raises_when_never_healthy():
    resp = MagicMock(); resp.status_code = 500
    proc = MagicMock(); proc.poll.return_value = None
    with patch("redteamrl.policies.vllm_client.subprocess.Popen", return_value=proc), \
         patch("redteamrl.policies.vllm_client.requests.get", return_value=resp), \
         patch("redteamrl.policies.vllm_client.time.sleep"):
        with pytest.raises(RuntimeError, match="never became healthy"):
            start_vllm_server("Qwen/Q", 8000, 0.4, timeout_s=4)
    proc.terminate.assert_called_once_with()
    proc.wait.assert_called_once_with(timeout=60)

def test_start_vllm_server_returns_handle():
    resp = MagicMock(); resp.status_code = 200
    sentinel = MagicMock()
    sentinel.poll.return_value = None
    with patch("redteamrl.policies.vllm_client.subprocess.Popen", return_value=sentinel), \
         patch("redteamrl.policies.vllm_client.requests.get", return_value=resp):
        proc = start_vllm_server("Qwen/Q", 8000, 0.4)
    assert proc is sentinel


def test_stop_vllm_server_terminates_and_waits():
    proc = MagicMock(); proc.poll.return_value = None
    stop_vllm_server(proc, timeout_s=7)
    proc.terminate.assert_called_once_with()
    proc.wait.assert_called_once_with(timeout=7)
    proc.kill.assert_not_called()


def test_stop_vllm_server_kills_after_timeout():
    proc = MagicMock(); proc.poll.return_value = None
    proc.wait.side_effect = [subprocess.TimeoutExpired("vllm", 7), 0]
    stop_vllm_server(proc, timeout_s=7)
    proc.terminate.assert_called_once_with()
    proc.kill.assert_called_once_with()
    assert proc.wait.call_count == 2


def test_seed_is_placed_in_the_request_payload(monkeypatch):
    captured = {}
    class R:
        status_code = 200
        def json(self): return {"choices": [{"message": {"content": "ok"}}]}
    import redteamrl.policies.vllm_client as vc
    monkeypatch.setattr(vc.requests, "post", lambda url, json, timeout: captured.update(p=json) or R())
    gen = make_vllm_generate("http://x", "m", seed=1234)
    gen("sys", [{"role": "user", "content": "hi"}])
    assert captured["p"]["seed"] == 1234


def test_generation_records_usage_and_termination_without_changing_text():
    session = MagicMock()
    session.post.return_value.status_code = 200
    session.post.return_value.json.return_value = {
        "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 10,
                  "prompt_tokens_details": {"cached_tokens": 80}},
    }
    records = []
    gen = make_vllm_generate("http://x", "m", max_tokens=10, label="action",
                            session=session, on_response=records.append)
    assert gen("sys", []) == "partial"
    assert records[0]["finish_reason"] == "length"
    assert records[0]["prompt_tokens"] == 120 and records[0]["cached_tokens"] == 80
    assert records[0]["completion_tokens"] == 10 and records[0]["label"] == "action"


def test_dead_server_fails_immediately_instead_of_billing_the_entire_startup_timeout():
    proc = MagicMock()
    proc.poll.return_value = 1
    with patch("redteamrl.policies.vllm_client.subprocess.Popen", return_value=proc), \
         patch("redteamrl.policies.vllm_client.time.sleep") as sleep, \
         patch("redteamrl.policies.vllm_client.requests.get") as get:
        with pytest.raises(RuntimeError, match="exited during startup"):
            start_vllm_server("m", 8000, 0.5, extra_args=["--enforce-eager"])
    sleep.assert_not_called()
    get.assert_not_called()


def test_serve_options_are_forwarded_and_existing_defaults_unchanged():
    from redteamrl.policies.vllm_client import build_serve_command
    default = build_serve_command("m", 8000, 0.5)
    args = ["--enforce-eager", "--revision", "abc"]
    assert build_serve_command("m", 8000, 0.5, extra_args=args) == default + args


def test_structured_action_schema_is_sent_without_affecting_unconstrained_calls():
    from redteamrl.harness.action_schema import ACTION_SCHEMA
    session = MagicMock()
    session.post.return_value.status_code = 200
    session.post.return_value.json.return_value = {"choices": [{"message": {"content": '{}'}}]}
    make_vllm_generate("http://x", "m", session=session, json_schema=ACTION_SCHEMA)("s", [])
    assert session.post.call_args.kwargs["json"]["structured_outputs"] == {"json": ACTION_SCHEMA}
    make_vllm_generate("http://x", "m", session=session)("s", [])
    assert "structured_outputs" not in session.post.call_args.kwargs["json"]


@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
def test_planner_keeps_reasoning_out_of_returned_scratchpad(reasoning_field):
    session = MagicMock()
    session.post.return_value.status_code = 200
    final = "## Main goal\ncontinue\n## Next action\ninspect"
    session.post.return_value.json.return_value = {
        "choices": [{"message": {"content": final, reasoning_field: "PRIVATE PLANNER NOTES"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 100},
    }
    records = []
    gen = make_vllm_generate("http://x", "m", enable_thinking=True, require_reasoning=True,
                            session=session, on_response=records.append)
    assert gen("sys", []) == final
    assert records[0]["reasoning"] == "PRIVATE PLANNER NOTES"
    assert records[0]["final_content"] == final
    assert records[0]["reasoning_present"] and records[0]["final_present"]
    assert session.post.call_args.kwargs["json"]["chat_template_kwargs"]["enable_thinking"]


@pytest.mark.parametrize("content,reasoning,finish", [
    (None, "still thinking", "length"),
    ("## Incomplete\n", "thinking", "length"),
    ("## Goal\ng", None, "stop"),
    ("", "thinking", "stop"),
    ("<think>private</think>## Goal\ng", "thinking", "stop"),
])
def test_planner_rejects_missing_or_unfinished_final_after_saving_telemetry(content, reasoning, finish):
    session = MagicMock()
    session.post.return_value.status_code = 200
    session.post.return_value.json.return_value = {
        "choices": [{"message": {"content": content, "reasoning": reasoning}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 100},
    }
    records = []
    gen = make_vllm_generate("http://x", "m", enable_thinking=True, require_reasoning=True,
                            session=session, on_response=records.append)
    with pytest.raises(RuntimeError, match="Thinking generation"):
        gen("sys", [])
    assert len(records) == 1 and records[0]["finish_reason"] == finish


def test_action_thinking_and_schema_share_request_but_only_json_reaches_agent():
    from redteamrl.harness.action_schema import ACTION_SCHEMA
    session = MagicMock()
    session.post.return_value.status_code = 200
    final = '{"kind":"tool_call","tool":"list_dir","args":{"path":"."}}'
    session.post.return_value.json.return_value = {
        "choices": [{"message": {"content": final, "reasoning": "Private check of plan."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 95},
    }
    records = []
    gen = make_vllm_generate("http://x", "m", label="action", enable_thinking=True,
                            require_reasoning=True, thinking_token_budget=256, max_tokens=1280,
                            json_schema=ACTION_SCHEMA, session=session, on_response=records.append)
    assert gen("sys", []) == final
    payload = session.post.call_args.kwargs["json"]
    assert payload["thinking_token_budget"] == 256
    assert payload["max_tokens"] == 1280
    assert payload["structured_outputs"] == {"json": ACTION_SCHEMA}
    assert records[0]["final_token_allowance"] == 1024
    assert records[0]["thinking_token_budget"] == 256


@pytest.mark.parametrize("thinking,budget,total", [(False, 100, 500), (True, 0, 500), (True, 500, 500)])
def test_invalid_thinking_budget_rejected_before_request(thinking, budget, total):
    with pytest.raises(ValueError, match="thinking_token_budget"):
        make_vllm_generate("http://x", "m", enable_thinking=thinking,
                           thinking_token_budget=budget, max_tokens=total)
