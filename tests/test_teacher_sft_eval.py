from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from redteamrl.sandbox.shell import LocalShellSandbox
from redteamrl.scratchpad.collection import probe_config, probe_jobs
from redteamrl.scratchpad.experiment import atomic_json
from teacher_sft.data import digest, write_jsonl
from teacher_sft.evaluate import compare, evaluate, evaluation_jobs, summarize_records


def prepared_fixture(root):
    cfg = probe_config(teacher="test", episodes=1, tier="T2", tag="test", source="test", env_seed_start=9000)
    task = probe_jobs(cfg)[0].spec.model_dump()
    # This fixture contains only evaluation cases; no train/author dataset is constructed.
    tasks = [dict(episode="ep", task=task, split="test", group="test-layout")]
    candidates = [dict(id="held:correction", episode="ep", split="test", kind="correction",
                       system="Test system", prompt="Actual visible test observation", tools=task["enabled_tools"],
                       action={"kind": "submit", "value": task["forbidden"][0]})]
    identity = dict(candidate_digest=digest(candidates), tasks_digest=digest(tasks))
    atomic_json(root / "manifest.json", dict(config=identity, identity=digest(identity)))
    write_jsonl(root / "candidates.jsonl", candidates)
    write_jsonl(root / "tasks.jsonl", tasks)
    return task


def test_paired_evaluation_and_correction_probe_never_pass_target_in_prompt(tmp_path):
    task = prepared_fixture(tmp_path / "prepared")
    jobs = evaluation_jobs(tmp_path / "prepared", n_generated=1)
    assert {j.kind for j in jobs} == {"generated", "reserved"}
    cfg = dict(n_generated=1, seed=2027, source="v1", max_turns=2, workers=2, adapter=None)
    prompts = []
    def generator(win):
        def make(job, callback):
            def call(system, messages):
                prompts.append(messages[0]["content"])
                callback(dict(finish_reason="stop", completion_tokens=10))
                value = job.spec.forbidden[0] if win else "wrong"
                return '## Plan\nTest only.\n' + json.dumps({"kind": "submit", "value": value})
            return call
        return make
    a = evaluate(tmp_path / "prepared", tmp_path / "base", cfg, generator(False), LocalShellSandbox)
    b = evaluate(tmp_path / "prepared", tmp_path / "tuned", cfg | {"adapter": "checkpoint"}, generator(True), LocalShellSandbox)
    assert a["overall"]["wins"] == 0
    assert b["overall"]["wins"] == len(jobs)
    assert a["correction_states"]["incorrect_submits"] == b["correction_states"]["correct_submits"] == 1
    assert "Actual visible test observation" in prompts
    assert all(task["forbidden"][0] not in p for p in prompts)
    result = compare(tmp_path / "base", tmp_path / "tuned", tmp_path / "comparison.json")
    assert result["gained"] == len(jobs) and result["lost"] == 0
    assert result["promoted"] is False
    assert result["paired_completed"] == len(jobs)
    b["config"]["max_turns"] = 3
    atomic_json(tmp_path / "tuned" / "summary.json", b)
    with pytest.raises(ValueError, match="configs"):
        compare(tmp_path / "base", tmp_path / "tuned", tmp_path / "comparison.json")


def test_eval_failure_has_own_denominator_and_resumes(tmp_path):
    prepared_fixture(tmp_path / "prepared")
    cfg = dict(n_generated=1, seed=2027, source="v1", max_turns=2, workers=2, adapter=None)
    def broken(*args):
        raise RuntimeError("simulated inference failure")
    report = evaluate(tmp_path / "prepared", tmp_path / "eval", cfg, broken, LocalShellSandbox)
    assert report["overall"]["completed"] == 0
    assert report["overall"]["failed"] == report["overall"]["requested"]
    assert report["overall"]["win_rate_completed"] is None
    assert evaluate(tmp_path / "prepared", tmp_path / "eval", cfg,
                    lambda *args: pytest.fail("must resume"), LocalShellSandbox) == report


def test_primary_invalid_not_hidden_by_successful_repair():
    record = dict(status="completed", task={"enabled_tools": ["read_file"], "forbidden": ["SECRET"]},
                  turns=[dict(scratchpad_completion="## Plan\nNo action", repaired=True)],
                  generations=[dict(finish_reason="length")],
                  result=dict(outcome="budget", leak=False, steps=[dict(turn=0,
                      action=dict(kind="submit", value="wrong"), returned="rejected")]))
    report = summarize_records([record])
    assert report["primary_invalid"] == report["repaired_turns"] == report["incorrect_submits"] == 1
    assert report["executed_invalid"] == 0 and report["length_finishes"] == 1


def test_execution_failures_and_defender_decisions_are_separate():
    record = dict(status="completed", task={"enabled_tools": ["read_file"], "forbidden": ["SECRET"]},
                  turns=[dict(scratchpad_completion='## Plan\nRead.\n{"kind":"tool_call","tool":"read_file","args":{"path":"x"}}')],
                  result=dict(outcome="budget", leak=False, steps=[dict(turn=0,
                      action=dict(kind="tool_call", tool="read_file", args={"path": "x"}),
                      call_decision={"verdict": "reject"}, execution=None)]))
    report = summarize_records([record])
    assert report["denied_calls"] == 1 and report["execution_errors"] == report["executed_invalid"] == 0
    record["result"]["steps"][0].update(call_decision={"verdict": "allow"}, execution={"status": "nonzero_exit"})
    report = summarize_records([record])
    assert report["execution_errors"] == 1 and report["denied_calls"] == 0


def test_modal_wiring_preserves_thinking_and_releases_server(tmp_path, monkeypatch):
    from scripts import scratchpad_sft as script
    from teacher_sft import author as author_module
    from redteamrl.policies import vllm_client
    events, closed = [], []
    monkeypatch.setattr(script, "ROOT", str(tmp_path))
    monkeypatch.setattr(script, "source_identity", lambda: "source")
    monkeypatch.setattr(script, "runs", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    monkeypatch.setattr(script, "cache", SimpleNamespace(reload=lambda: None))
    monkeypatch.setattr(script, "start_server", lambda cfg: events.append(cfg) or "process")
    monkeypatch.setattr(vllm_client, "stop_vllm_server", closed.append)
    def generate(config, cb):
        def call(system, messages):
            cb(dict(reasoning="Student reasoning", final_content="Final", finish_reason="stop"))
        return call
    monkeypatch.setattr(script, "generate_for", generate)
    def author(prepared, out, generate, config, **kwargs):
        assert config["enable_thinking"] is True
        assert generate("sys", "prompt")[:2] == ("Student reasoning", "Final")
        raise RuntimeError("test authoring failure")
    monkeypatch.setattr(author_module, "author_dataset", author)
    with pytest.raises(RuntimeError, match="authoring failure"):
        script.author.local("test", "revision", 8)
    assert closed == ["process"] and events[0]["max_candidates"] == 8
    assert events[0]["thinking_token_budget"] < events[0]["max_tokens"]


def test_modal_cli_validates_before_dispatch(monkeypatch):
    from scripts import scratchpad_sft as script
    monkeypatch.setattr(script, "train", SimpleNamespace(remote=lambda *a: pytest.fail("dispatched")))
    for kwargs in ({"tag": "../escape"}, {"max_steps": 0}, {"arm": "base"}):
        with pytest.raises(ValueError):
            script.main(stage="train", **kwargs)
