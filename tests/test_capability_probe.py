"""Offline: capability-probe CLI guards and GPU-serving wiring (no model, no network)."""
import importlib.util

from unittest.mock import MagicMock

import pytest


def load_script():
    spec = importlib.util.spec_from_file_location("capability_script", "scripts/capability_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kwargs", [{"tag": "../bad"}, {"tensor_parallel": 0}, {"tensor_parallel": 9},
                                    {"thinking_token_budget": 8192}, {"max_tokens": 9000},
                                    {"episodes": 0}, {"workers": 0}])
def test_cli_rejects_before_remote_dispatch(kwargs):
    module = load_script()
    with pytest.raises(ValueError):
        module.main(**kwargs)


def test_cli_dispatches_with_threaded_args(monkeypatch):
    module = load_script()
    sent = {}
    monkeypatch.setattr(module.probe, "remote", lambda *a: sent.setdefault("args", a))
    module.main(tag="cap-test", model="Qwen/Qwen3.8-27B", tensor_parallel=1)
    assert sent["args"][0] == "cap-test" and sent["args"][1] == "Qwen/Qwen3.8-27B"


def test_probe_gpu_wiring_serves_and_builds_primary_and_forcer(monkeypatch):
    module = load_script()
    for vol in (module.runs, module.cache):
        monkeypatch.setattr(vol, "reload", lambda: None)
        monkeypatch.setattr(vol, "commit", lambda: None)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: "/cache")
    monkeypatch.setattr("redteamrl.scratchpad.experiment.source_digest", lambda *a, **k: "src")
    monkeypatch.setattr("redteamrl.scratchpad.experiment.atomic_json", lambda *a, **k: None)
    start, stop, make = MagicMock(return_value="server"), MagicMock(), MagicMock()
    monkeypatch.setattr("redteamrl.policies.vllm_client.start_vllm_server", start)
    monkeypatch.setattr("redteamrl.policies.vllm_client.stop_vllm_server", stop)
    monkeypatch.setattr("redteamrl.policies.vllm_client.make_vllm_generate", make)
    monkeypatch.setattr("redteamrl.scratchpad.routes.route_diversity", lambda recs: dict(
        completed=0, wins=0, routes_attempted={}, routes_won={}, distinct_winning_routes=0,
        winning_routes=[], novel_other_wins=0, per_episode=[]))

    class Job:
        key = "ep-T2-000"

    # Exercise make_generate once (building primary + forcer) without running a real episode.
    def fake_episode(job, config, make_generate, make_sandbox):
        make_generate(lambda event: None)
        return {"status": "completed", "result": {"leak": False, "steps": []}}

    def fake_run_collection(config, out, execute, **kw):
        execute(Job())
        return {"summary": {"T2": dict(n_completed=1, n_requested=1, n_failed=0, win_rate=0.0,
                                       recovery_wins=0, recovery_eligible=0), "gate_status": "INCONCLUSIVE"}}

    monkeypatch.setattr("redteamrl.scratchpad.collection.execute_teacher_episode", fake_episode)
    monkeypatch.setattr("redteamrl.scratchpad.collection.run_collection", fake_run_collection)

    module.probe.local("cap-test", tensor_parallel=1)

    assert start.call_args.args[0] == "Qwen/Qwen3.8-27B"
    extra = start.call_args.kwargs["extra_args"]
    assert "--reasoning-parser" in extra and "--revision" in extra
    assert extra[extra.index("--tensor-parallel-size") + 1] == "1"
    assert "--enforce-eager" not in extra  # CUDA graphs on by default; eager is opt-in fallback
    primary, forcer = make.call_args_list  # primary (thinking-ON) then the action-forcer (thinking-OFF)
    assert primary.kwargs["require_reasoning"] and primary.kwargs["enable_thinking"]
    assert primary.kwargs["thinking_token_budget"] == 6144
    assert not forcer.kwargs["require_reasoning"] and not forcer.kwargs["enable_thinking"]
    assert forcer.kwargs["max_tokens"] == 512
    # Both calls get the generous per-request timeout so a long 27B thinking turn is not abandoned.
    assert primary.kwargs["timeout"] == 1200.0 and forcer.kwargs["timeout"] == 1200.0
    stop.assert_called_once_with("server")
