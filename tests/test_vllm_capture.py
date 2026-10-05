import pytest

from redteamrl.train.capture import Example, VLLMCapturingGenerate


class _Tok:
    def apply_chat_template(self, chat, add_generation_prompt, enable_thinking, tokenize):
        assert tokenize is False          # we send text; vLLM returns the ids it actually used
        return "|".join(m["content"] for m in chat)


def _post(payload, seen=None):
    def post(url, json, timeout):
        if seen is not None:
            seen.append((url, json))

        class _R:
            status_code = 200

            @staticmethod
            def json():
                return payload
        return _R()
    return post


def _cap(payload, seen=None):
    return VLLMCapturingGenerate(
        base_url="http://localhost:8001", model="defender", tokenizer=_Tok(),
        temperature=0.7, max_new_tokens=64, post=_post(payload, seen),
    )


_OK = {"choices": [{"text": " ok", "prompt_token_ids": [1, 2, 3], "token_ids": [7, 8]}]}


def test_captures_the_exact_ids_vllm_sampled():
    cap = _cap(_OK)
    with cap.episode_capture() as buffer:
        out = cap("SYS", [{"role": "user", "content": "hello"}])

    assert out == " ok"
    assert len(buffer) == 1
    example = buffer[0]
    assert isinstance(example, Example)
    assert example.prompt_ids == [1, 2, 3]
    assert example.completion_ids == [7, 8]


def test_requests_token_ids_and_disables_extra_special_tokens():
    seen = []
    cap = _cap(_OK, seen)
    cap("SYS", [{"role": "user", "content": "hello"}])

    url, body = seen[0]
    assert url.endswith("/v1/completions")
    assert body["return_token_ids"] is True
    # The chat template already emits BOS; letting the server add another would shift every id
    # relative to the prompt HF re-scores during the update.
    assert body["add_special_tokens"] is False
    assert body["model"] == "defender"
    assert body["prompt"] == "SYS|hello"


def test_episode_capture_is_thread_scoped_like_the_hf_path():
    cap = _cap(_OK)
    with cap.episode_capture() as first:
        cap("SYS", [{"role": "user", "content": "a"}])
    with cap.episode_capture() as second:
        cap("SYS", [{"role": "user", "content": "b"}])

    assert len(first) == len(second) == 1
    assert first[0] is not second[0]


def test_a_response_without_ids_raises_rather_than_training_on_a_guess():
    cap = _cap({"choices": [{"text": "ok"}]})
    with pytest.raises(ValueError, match="return_token_ids"):
        cap("SYS", [{"role": "user", "content": "hello"}])
