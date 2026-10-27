"""Exercise the actual Modal entrypoint locally: no network, provider, volume or GPU use."""
import json
import threading
from types import SimpleNamespace

import pytest

from scripts import gate0_probe as script
from redteamrl.sandbox.shell import LocalShellSandbox


def test_handler_wires_episode_local_telemetry_and_safe_resume(tmp_path, monkeypatch):
    from redteamrl.policies import openrouter_client
    from redteamrl.sandbox import modal_shell
    from redteamrl.scratchpad import experiment
    events, sessions = [], []
    coordinator = threading.get_ident()
    monkeypatch.setattr(script, "RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(script, "runs", SimpleNamespace(
        reload=lambda: None, commit=lambda: events.append(threading.get_ident())))
    monkeypatch.setattr(experiment, "source_digest", lambda *a: "source-v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(modal_shell, "ModalShellSandbox", lambda *a, **kw: LocalShellSandbox())

    def generate(model, key, **kw):
        assert model == "test" and key == "fake-key"
        assert kw["max_tokens"] == 8192
        sessions.append(kw["session"])
        def call(system, messages):
            assert "Final answer: ONLY one JSON" not in messages[0]["content"]
            kw["on_response"]({"label":"teacher", "finish_reason":"stop", "prompt_tokens":10,
                                "completion_tokens":20, "response_id":f"resp-{id(kw['session'])}"})
            return '## Plan\nInspect.\n{"kind":"tool_call","tool":"list_dir","args":{"path":"."}}'
        return call
    monkeypatch.setattr(openrouter_client, "make_openrouter_generate", generate)
    report = script.probe.local("test", episodes=2, tag="smoke", max_turns=2, workers=2)
    assert report["n_completed"] == 2 and report["n_failed"] == 0
    assert report["summary"]["gate_status"] == "NO-GO"
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert set(events) == {coordinator}
    records = [json.loads(p.read_text()) for p in (tmp_path / "smoke").glob("ep-*.json")]
    assert len(records) == 2
    assert all([g["turn"] for g in r["generations"]] == [0,1] for r in records)
    assert len({r["generations"][0]["response_id"] for r in records}) == 2
    assert script.probe.local("test", episodes=2, tag="smoke", max_turns=2, workers=2) == report
    assert len(sessions) == 2
    monkeypatch.setattr(experiment, "source_digest", lambda *a: "source-v2")
    with pytest.raises(ValueError, match="fresh --tag"):
        script.probe.local("test", episodes=2, tag="smoke", max_turns=2, workers=2)
    assert len(sessions) == 2


def test_cli_rejects_unsupported_tier_before_dispatch(monkeypatch):
    monkeypatch.setattr(script, "probe", SimpleNamespace(remote=lambda *a: pytest.fail("dispatched")))
    with pytest.raises(ValueError, match="T2 only"):
        script.main(tier="T1")
