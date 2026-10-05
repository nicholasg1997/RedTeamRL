import pytest

from redteamrl.policies.vllm_client import (
    build_serve_command, parse_token_ids,
)


def test_serve_command_gains_lora_flags_only_when_requested():
    plain = build_serve_command("Qwen/Qwen3-4B", 8001, 0.15, max_model_len=16384)
    assert "--enable-lora" not in plain

    with_lora = build_serve_command(
        "Qwen/Qwen3-4B", 8001, 0.15, max_model_len=16384,
        lora_name="defender", lora_path="/tmp/a", max_lora_rank=16,
    )
    assert "--enable-lora" in with_lora
    assert "defender=/tmp/a" in with_lora
    assert with_lora[with_lora.index("--max-lora-rank") + 1] == "16"


def test_lora_path_requires_a_name():
    with pytest.raises(ValueError, match="lora_name"):
        build_serve_command("m", 1, 0.1, lora_path="/tmp/a")


def _payload(choice, **top):
    return {"choices": [choice], **top}


def test_reads_token_ids_from_the_choice():
    text, prompt_ids, completion_ids = parse_token_ids(
        _payload({"text": "hi", "prompt_token_ids": [1, 2], "token_ids": [3, 4]})
    )
    assert (text, prompt_ids, completion_ids) == ("hi", [1, 2], [3, 4])


def test_reads_token_ids_hoisted_to_the_top_level():
    # vLLM has moved these between the choice and the payload across versions; accept both rather
    # than silently falling back to retokenization, which would change what the gradient scores.
    text, prompt_ids, completion_ids = parse_token_ids(
        _payload({"text": "hi", "token_ids": [3, 4]}, prompt_token_ids=[1, 2])
    )
    assert (prompt_ids, completion_ids) == ([1, 2], [3, 4])


def test_alternate_completion_id_field_name():
    text, prompt_ids, completion_ids = parse_token_ids(
        _payload({"text": "hi", "prompt_token_ids": [1], "completion_token_ids": [9, 8]})
    )
    assert completion_ids == [9, 8]


def test_missing_ids_raise_instead_of_guessing():
    # Silently retokenizing would compute log-probs for a DIFFERENT token sequence than vLLM
    # sampled, corrupting the importance ratio with no visible error. Fail loudly.
    with pytest.raises(ValueError, match="return_token_ids"):
        parse_token_ids(_payload({"text": "hi"}))


def test_empty_completion_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        parse_token_ids(_payload({"text": "", "prompt_token_ids": [1], "token_ids": []}))
