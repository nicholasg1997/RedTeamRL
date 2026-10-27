import pytest
from unittest.mock import MagicMock
from redteamrl.policies.openrouter_client import make_openrouter_generate


def _resp(status=200, **body):
    r = MagicMock(); r.status_code = status
    r.json.return_value = body
    r.text = str(body)
    return r


def test_builds_authed_openai_payload_and_returns_content():
    session = MagicMock()
    session.post.return_value = _resp(choices=[{"message": {"content": "notes\n{\"kind\":\"submit\",\"value\":\"x\"}"}}])
    gen = make_openrouter_generate("nvidia/nemotron-3-ultra-550b-a55b:free", api_key="sk-or-K",
                                   session=session, temperature=0.7)
    out = gen("SYS", [{"role": "user", "content": "hi"}])
    assert out.endswith('"value":"x"}')
    url = session.post.call_args.args[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    headers = session.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer sk-or-K"
    payload = session.post.call_args.kwargs["json"]
    assert payload["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert payload["messages"][0] == {"role": "system", "content": "SYS"}


def test_non_200_surfaces_body_but_masks_key():
    session = MagicMock()
    session.post.return_value = _resp(status=402, error="insufficient credits")
    gen = make_openrouter_generate("m", api_key="sk-or-SECRET", session=session)
    with pytest.raises(RuntimeError) as e:
        gen("SYS", [{"role": "user", "content": "hi"}])
    assert "insufficient credits" in str(e.value)
    assert "sk-or-SECRET" not in str(e.value)


def test_reasoning_field_is_dropped_only_content_returned():
    # Reasoning models return the think trace separately; the attacker consumes the ANSWER only.
    session = MagicMock()
    session.post.return_value = _resp(choices=[{"message": {"content": "the answer", "reasoning": "long cot"}}])
    gen = make_openrouter_generate("m", api_key="k", session=session)
    assert gen("SYS", []) == "the answer"


def test_empty_content_degrades_to_empty_string_not_crash():
    # A reasoning model can spend its whole budget thinking and emit no answer. One such turn must
    # NOT crash the episode -- the agent turns "" into a consumed/invalid turn and the episode goes on.
    session = MagicMock()
    session.post.return_value = _resp(choices=[{"message": {"content": None, "reasoning": "only thought"}}])
    gen = make_openrouter_generate("m", api_key="k", session=session)
    assert gen("SYS", []) == ""


def test_default_token_budget_is_generous_for_reasoning_models():
    session = MagicMock()
    session.post.return_value = _resp(choices=[{"message": {"content": "ok"}}])
    gen = make_openrouter_generate("m", api_key="k", session=session)
    gen("SYS", [])
    assert session.post.call_args.kwargs["json"]["max_tokens"] >= 8192


def test_rate_limit_retries_then_succeeds():
    session = MagicMock()
    session.post.side_effect = [_resp(status=429, error="rate limited"),
                                _resp(choices=[{"message": {"content": "ok"}}])]
    gen = make_openrouter_generate("m", api_key="k", session=session, max_retries=2, backoff_s=0.0)
    assert gen("SYS", []) == "ok"
    assert session.post.call_count == 2


def test_response_callback_keeps_usage_finish_and_identity_without_reasoning_or_credentials():
    calls = []
    session = MagicMock()
    session.post.return_value = _resp(id="resp1", model="served", provider="p", usage={
        "prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50,
        "completion_tokens_details": {"reasoning_tokens": 25}, "cost": 0.001}, choices=[{
        "finish_reason": "length", "native_finish_reason": "max_tokens",
        "message": {"content": None, "reasoning": "PRIVATE REASONING"}}])
    gen = make_openrouter_generate("requested", "API_SECRET", session=session, on_response=calls.append)
    assert gen("PRIVATE SYSTEM", []) == ""
    [meta] = calls
    assert meta["finish_reason"] == "length" and meta["native_finish_reason"] == "max_tokens"
    assert meta["reasoning_tokens"] == 25 and meta["cost"] == 0.001
    assert meta["response_id"] == "resp1" and meta["requested_model"] == "requested"
    assert meta["model"] == "served" and meta["reasoning_present"] and not meta["final_present"]
    assert meta["latency_s"] >= 0
    assert all(private not in str(meta) for private in ["API_SECRET", "PRIVATE REASONING", "PRIVATE SYSTEM"])


def test_retry_attempts_share_call_id_and_missing_usage_stays_unknown():
    calls = []
    session = MagicMock()
    session.post.side_effect = [_resp(status=429), _resp(choices=[{"message":{"content":"ok"}}]),
                                _resp(choices=[{"message":{"content":"again"}}])]
    gen = make_openrouter_generate("m", "SECRET", session=session, on_response=calls.append, backoff_s=0)
    assert gen("s", []) == "ok"
    assert gen("s", []) == "again"
    assert [(m["call_id"], m["attempt"]) for m in calls] == [(1,1), (1,2), (2,1)]
    assert calls[0]["error_type"] == "RateLimit"
    assert calls[1]["finish_reason"] is None and calls[1]["prompt_tokens"] is None


def test_provider_refusal_is_explicit_not_inferred_from_empty_content():
    calls = []
    session = MagicMock()
    session.post.return_value = _resp(choices=[{"finish_reason":"content_filter",
                                              "message":{"content":None, "refusal":"blocked"}}])
    gen = make_openrouter_generate("m", "SECRET", session=session, on_response=calls.append)
    assert gen("s", []) == ""
    assert calls[0]["refusal_present"] is True


@pytest.mark.parametrize("status", [200, 503])
def test_provider_errors_are_recorded_and_do_not_become_empty_actions(status):
    calls = []
    session = MagicMock()
    session.post.return_value = _resp(status=status, error={"message":"API_SECRET"})
    gen = make_openrouter_generate("m", "API_SECRET", session=session, on_response=calls.append)
    with pytest.raises(RuntimeError) as error:
        gen("s", [])
    assert "API_SECRET" not in str(error.value)
    assert calls[0]["error_type"] in {"ProviderError", "HTTPError"}


def test_invalid_response_json_is_recorded_as_failure():
    calls = []
    session = MagicMock()
    session.post.return_value = _resp()
    session.post.return_value.json.side_effect = ValueError("invalid JSON")
    gen = make_openrouter_generate("m", "SECRET", session=session, on_response=calls.append)
    with pytest.raises(RuntimeError, match="completion JSON"):
        gen("s", [])
    assert calls[0]["error_type"] == "ResponseSchemaError"


def test_missing_choices_is_a_provider_failure_not_a_model_empty_response():
    calls = []
    session = MagicMock()
    session.post.return_value = _resp(choices=[])
    gen = make_openrouter_generate("m", "SECRET", session=session, on_response=calls.append)
    with pytest.raises(RuntimeError, match="provider error"):
        gen("s", [])
    assert calls[0]["error_type"] == "MissingChoices"
