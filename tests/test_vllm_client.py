import pytest
from unittest.mock import patch, MagicMock
from redteamrl.policies.vllm_client import make_vllm_generate, start_vllm_server

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
        start_vllm_server("Qwen/Q", 8000, 0.4, max_model_len=4096, max_num_seqs=16)
    cmd = popen.call_args.args[0]
    assert cmd[:3] == ["vllm", "serve", "Qwen/Q"]
    assert "0.4" in cmd and "4096" in cmd and "8000" in cmd
    assert "--max-num-seqs" in cmd and "16" in cmd


def test_start_vllm_server_raises_when_never_healthy():
    resp = MagicMock(); resp.status_code = 500
    with patch("redteamrl.policies.vllm_client.subprocess.Popen"), \
         patch("redteamrl.policies.vllm_client.requests.get", return_value=resp), \
         patch("redteamrl.policies.vllm_client.time.sleep"):
        with pytest.raises(RuntimeError, match="never became healthy"):
            start_vllm_server("Qwen/Q", 8000, 0.4, timeout_s=4)