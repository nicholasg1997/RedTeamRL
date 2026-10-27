"""Exercise the actual Modal handler locally with fake inference, volumes and local sandboxes."""
import json
import pytest
from types import SimpleNamespace

from redteamrl.sandbox.shell import LocalShellSandbox
from redteamrl.scratchpad.tiers import Tier
from redteamrl.types import DefenderDecision
from redteamrl.verify.defenders import OracleRedactorDefender
from scripts import scratchpad_poc as script


class RejectAll(OracleRedactorDefender):
    def adjudicate_call(self, conversation, action):
        return DefenderDecision(verdict="reject")


@pytest.mark.parametrize("planner_thinking,action_thinking", [(False, False), (True, False), (True, True), (False, True)])
def test_handler_gates_before_attacks_skips_invalid_tier_and_resumes(tmp_path, monkeypatch, planner_thinking, action_thinking):
    from redteamrl.policies import vllm_client
    from redteamrl.sandbox import modal_shell
    from redteamrl.scratchpad import tiers
    events = []
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: events.append("commit"))
    for name in ("hf_cache", "runs", "inference_cache"):
        monkeypatch.setattr(script, name, volume)
    monkeypatch.setattr(script, "RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setattr(modal_shell, "ModalShellSandbox", lambda *a, **kw: LocalShellSandbox())
    monkeypatch.setattr(tiers, "make_tiers", lambda **kw: [
        Tier("T0", lambda spec, gen: OracleRedactorDefender(spec.forbidden)),
        Tier("T1", lambda spec, gen: RejectAll(spec.forbidden)),
        Tier("T2", lambda spec, gen: OracleRedactorDefender(spec.forbidden)),
    ])

    def start(model, *args, **kw):
        events.append(model)
        assert "--enforce-eager" in kw["extra_args"]
        assert "--revision" in kw["extra_args"]
        assert ("--reasoning-parser" in kw["extra_args"]) == ((planner_thinking or action_thinking) and model == script.ATTACKER_MODEL)
        return model

    monkeypatch.setattr(vllm_client, "start_vllm_server", start)
    monkeypatch.setattr(vllm_client, "stop_vllm_server", lambda model: events.append(f"stop:{model}"))

    def generate(*args, **kw):
        assert (kw.get("json_schema") is not None) == (kw["label"] == "action")
        thinking = (planner_thinking and kw["label"] == "scratchpad") or (action_thinking and kw["label"] == "action")
        assert kw["enable_thinking"] == kw["require_reasoning"] == thinking
        if thinking:
            assert kw["thinking_token_budget"] == (1536 if kw["label"] == "scratchpad" else 256)
            assert kw["max_tokens"] - kw["thinking_token_budget"] == (768 if kw["label"] == "scratchpad" else 1024)
        else:
            assert kw["thinking_token_budget"] is None
        def call(system, messages):
            kw["on_response"]({"label": kw["label"], "finish_reason": "stop",
                               "prompt_tokens": 50, "completion_tokens": 10})
            if kw["label"] == "scratchpad":
                return "## Main goal\nrecover target\n## Next ideas\ninspect"
            return '{"kind":"tool_call","tool":"list_dir","args":{"path":"."}}'
        return call

    monkeypatch.setattr(vllm_client, "make_vllm_generate", generate)
    config = script.make_config(max_turns=2, planner_thinking=planner_thinking, action_thinking=action_thinking)
    config["model_revisions"] = {script.ATTACKER_MODEL: "a", script.DEFENDER_MODEL: "b"}
    summary = script.run.local(config, "test")
    assert summary["gates"]["T0"]["valid"]
    assert not summary["gates"]["T1"]["valid"]
    assert summary["arms"]["full/T1"]["n_skipped"] == 2
    assert summary["arms"]["full/T0"]["n_completed"] == 2
    def_start = events.index(script.DEFENDER_MODEL)
    atk_start = events.index(script.ATTACKER_MODEL)
    assert events[def_start:atk_start].count("commit") == 25  # 24 benign records + gate report
    assert [e for e in events if e.startswith("stop:")] == [
        f"stop:{script.ATTACKER_MODEL}", f"stop:{script.DEFENDER_MODEL}"]
    assert json.loads((tmp_path / "test" / "summary.json").read_text())["arms"] == summary["arms"]
    events.clear()
    assert script.run.local(config, "test") == summary
    assert not events  # no inference startup, downloads or recomputation on a completed resume


def test_cpu_preparation_pins_models_and_completed_resume_needs_no_download(tmp_path, monkeypatch):
    import huggingface_hub
    import threading
    from redteamrl.scratchpad import experiment
    from redteamrl.sandbox import modal_shell
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    monkeypatch.setattr(script, "hf_cache", volume)
    monkeypatch.setattr(script, "runs", volume)
    monkeypatch.setattr(script, "RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(experiment, "source_digest", lambda *a: "source-1")
    monkeypatch.setattr(modal_shell, "ModalShellSandbox", lambda *a, **kw: LocalShellSandbox())
    downloads = []
    events = []
    preparation_thread = threading.get_ident()
    monkeypatch.setattr(script, "hf_cache", SimpleNamespace(
        reload=lambda: None, commit=lambda: events.append("cache_commit")))

    def download(model, revision, **kw):
        # Outer concurrent downloads caused a real tqdm lock-cleanup failure on Modal.
        assert threading.get_ident() == preparation_thread
        events.append(model)
        downloads.append((model, revision))
        return f"/cache/models/snapshots/{model.rsplit('/', 1)[-1]}-sha"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    prepared = script.prepare.local(script.make_config(), "test")
    assert prepared["summary"] is None
    assert prepared["config"]["model_revisions"][script.ATTACKER_MODEL] == "Qwen3-8B-sha"
    assert len(downloads) == 2
    assert events == [script.ATTACKER_MODEL, "cache_commit",
                      script.DEFENDER_MODEL, "cache_commit"]
    (tmp_path / "test" / "summary.json").write_text('{"saved": true}')
    downloads.clear()
    resumed = script.prepare.local(script.make_config(), "test")
    assert resumed["summary"] == {"saved": True}
    assert downloads == []
    import pytest
    with pytest.raises(ValueError, match="fresh --tag"):
        script.prepare.local(script.make_config(mode="both"), "test")


def test_real_mixed_benign_twin_exercises_selective_redaction():
    from redteamrl.scratchpad.experiment import plan_jobs
    from redteamrl.scratchpad.gate import run_benign_gate
    from redteamrl.scratchpad.tiers import make_tiers
    jobs, _ = plan_jobs()
    spec = next(j.spec for j in jobs if j.spec.id.endswith("-mixed-read"))
    path = spec.client_actions[0].args["path"]
    assert any(spec.forbidden[0] in f.content and spec.required[0] in f.content
               for f in spec.seed_files if f.path == path)
    oracle = next(t for t in make_tiers() if t.name == "T2")
    [result] = run_benign_gate(oracle, [spec], None, LocalShellSandbox)
    assert result.complete and not result.leak and result.effective_redactions > 0


def test_planner_options_are_bounded_and_manifest_identifiable():
    base = script.make_config()
    thinking = script.make_config(planner_thinking=True)
    assert not base["planner_thinking"] and base["scratchpad_max_tokens"] == 768
    assert thinking["planner_thinking"] and thinking["scratchpad_max_tokens"] == 2304
    assert script.make_config(planner_thinking=True, planner_max_tokens=3072)["scratchpad_max_tokens"] == 3072
    with pytest.raises(ValueError, match="requires a scratchpad"):
        script.make_config(mode="tried-only", planner_thinking=True)
    with pytest.raises(ValueError, match="planner-max-tokens"):
        script.make_config(planner_max_tokens=9000)


def test_two_stage_budget_validation():
    config = script.make_config(planner_thinking=True, action_thinking=True)
    assert config["planner_thinking_budget"] == 1536
    assert config["action_thinking_budget"] == 256
    assert config["action_max_tokens"] == 1280
    with pytest.raises(ValueError, match="leave at least"):
        script.make_config(planner_thinking=True, planner_max_tokens=1600)
    with pytest.raises(ValueError, match="action-thinking-budget"):
        script.make_config(action_thinking_budget=2048)
    with pytest.raises(ValueError, match="planner-thinking-budget"):
        script.make_config(planner_thinking_budget=-1)
