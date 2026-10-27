import json
from pathlib import Path
import threading

import pytest

from redteamrl.harness.episode import EpisodeResult
from redteamrl.scratchpad.experiment import (
    TIERS, atomic_json, ensure_manifest, gate_summary, plan_jobs, run_jobs, summarize, validate_options,
)


def result(complete=True, leak=False):
    return EpisodeResult(outcome="leak" if leak else "timeout", n_turns=1, complete=complete,
                         leak=leak, attacker_reward=0, defender_reward=0, steps=[]).model_dump()


def test_default_plan_is_small_and_both_arms_are_matched():
    gates, attacks = plan_jobs()
    assert len(gates) == 24 and len(attacks) == 6
    gates, attacks = plan_jobs(mode="both")
    assert len(gates) == 24 and len(attacks) == 12
    for task_id in {j.spec.id for j in attacks}:
        group = [j for j in attacks if j.spec.id == task_id]
        assert len(group) == 6
        assert len({j.generation_seed for j in group}) == 1
        assert all(j.spec == group[0].spec for j in group)
    for gate in gates:
        attack = next(j for j in attacks if j.spec.forbidden == gate.spec.forbidden)
        assert gate.spec.seed_files == attack.spec.seed_files
        assert gate.spec.forbidden[0] not in (gate.spec.request or "")


@pytest.mark.parametrize("change", [{"n_episodes": 4}, {"n_episodes": 0}, {"tag": "../escape"},
                                     {"workers": 0}, {"benign_envs": 0}, {"mode": "other"}])
def test_bad_options_fail_before_remote_work(change):
    args = dict(n_episodes=6, max_turns=12, tag="smoke-v1", mode="full", workers=8, benign_envs=2)
    args.update(change)
    with pytest.raises(ValueError):
        validate_options(**args)


def test_manifest_allows_identical_resume_but_rejects_changed_configuration(tmp_path):
    config = {"mode": "full", "canary_revision": 201, "source": "a"}
    a = ensure_manifest(tmp_path, config)
    assert ensure_manifest(tmp_path, config) == a
    for key, value in [("mode", "both"), ("canary_revision", 202), ("source", "b")]:
        with pytest.raises(ValueError, match="fresh --tag"):
            ensure_manifest(tmp_path, config | {key: value})


def test_legacy_unidentified_results_are_never_reused(tmp_path):
    atomic_json(tmp_path / "ep-old.json", {"result": result()})
    with pytest.raises(ValueError, match="no manifest"):
        ensure_manifest(tmp_path, {})


def test_workers_preserve_failures_and_coordinator_alone_commits(tmp_path):
    _, jobs = plan_jobs()
    main_thread = threading.get_ident()
    commits = []

    def execute(job):
        assert threading.get_ident() != main_thread
        if job.tier == "T1":
            raise RuntimeError("request failed")
        return {"status": "completed", "result": result()}

    records = run_jobs(jobs, execute, tmp_path, workers=3,
                       commit=lambda: commits.append(threading.get_ident()))
    assert commits == [main_thread] * 6
    assert sum(r["status"] == "failed" for r in records) == 2
    assert all(Path(tmp_path, f"{j.key}.json").exists() for j in jobs)
    assert run_jobs(jobs, lambda j: pytest.fail("resume reran an episode"), tmp_path,
                    workers=3) == records
    retried = []
    run_jobs(jobs, lambda j: retried.append(j.key) or {"status": "completed", "result": result()},
             tmp_path, workers=2, retry_failed=True)
    assert set(retried) == {j.key for j in jobs if j.tier == "T1"}


def test_commit_failure_is_fatal_not_an_episode_failure(tmp_path):
    _, jobs = plan_jobs()
    def broken_commit():
        raise OSError("volume unavailable")
    with pytest.raises(OSError, match="volume unavailable"):
        run_jobs(jobs[:1], lambda j: {"status": "completed", "result": result()},
                 tmp_path, workers=1, commit=broken_commit)
    assert json.loads((tmp_path / f"{jobs[0].key}.json").read_text())["status"] == "completed"


def test_report_does_not_present_skips_or_missing_episodes_as_zero_win_rate():
    gate_jobs, jobs = plan_jobs()
    benign = [{**j.metadata(), "status": "completed", "result": result(j.tier != "T1")}
              for j in gate_jobs]
    gates = gate_summary(benign)
    records = [{**j.metadata(), "status": "skipped"} if j.tier == "T1" else
               {**j.metadata(), "status": "completed", "result": result(leak=True)} for j in jobs]
    records[0] = {**jobs[0].metadata(), "status": "failed"}
    rows = summarize(gates, records)["arms"]
    assert rows["full/T1"]["win_rate"] is None and rows["full/T1"]["n_skipped"] == 2
    assert rows["full/T0"]["win_rate"] is None and rows["full/T0"]["n_failed"] == 1
    assert rows["full/T0"]["win_rate_completed"] == 1.0
    assert rows["full/T2"]["win_rate"] == 1.0
    assert rows["full/T2"]["recovery_rate"] is None


def test_length_finishes_are_reported_and_block_operational_ok():
    gate_jobs, jobs = plan_jobs()
    gates = gate_summary([{**j.metadata(), "status": "completed", "result": result()} for j in gate_jobs])
    records = [{**j.metadata(), "status": "completed", "result": result(), "generations": [
        {"label": "action", "finish_reason": "length", "prompt_tokens": 400,
         "completion_tokens": 100, "latency_s": 1.0}]} for j in jobs]
    row = summarize(gates, records)["arms"]["full/T0"]
    assert not row["operational_ok"]
    assert row["generations"]["action"]["length_finishes"] == 2
    assert row["generations"]["action"]["completion_tokens"] == 200


def test_skipped_episodes_can_be_retried_after_gate_recovers(tmp_path):
    _, jobs = plan_jobs()
    run_jobs(jobs[:1], lambda j: {"status": "skipped"}, tmp_path, workers=1)
    rows = run_jobs(jobs[:1], lambda j: {"status": "completed", "result": result()},
                    tmp_path, workers=1, retry_failed=True)
    assert rows[0]["status"] == "completed"


def test_generation_summary_reports_reasoning_and_missing_final():
    from redteamrl.scratchpad.experiment import generation_summary
    result = generation_summary([{"generations": [{
        "label": "scratchpad", "thinking_enabled": True, "reasoning_present": True,
        "final_present": False, "finish_reason": "length", "prompt_tokens": 30,
        "completion_tokens": 2048,
    }]}])["scratchpad"]
    assert result["thinking_calls"] == result["missing_final"] == result["length_finishes"] == 1
    assert result["missing_reasoning"] == 0
