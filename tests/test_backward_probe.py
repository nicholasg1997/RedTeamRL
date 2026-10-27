"""Offline: only controlled fixture commands execute in the local test sandbox."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shlex
from unittest.mock import MagicMock

import pytest

from backward_probe.data import load_prepared, prepare, select_sources
from backward_probe.runner import assisted_prompt, build_jobs, execute, generation_config, run, summarize
from backward_probe.state import ReplayMismatch, make_cases, predecision_scratchpad, recorded_state, replay
from redteamrl.harness.restart import sandbox_content_hash
from redteamrl.sandbox.shell import LocalShellSandbox
from redteamrl.scratchpad.collection import execute_teacher_episode, probe_config, probe_jobs, run_collection
from redteamrl.scratchpad.experiment import atomic_json
from teacher_sft.data import digest, split_for


class Sandbox(LocalShellSandbox):
    def state_hash(self):
        return sandbox_content_hash(self.root)


@pytest.fixture
def source():
    cfg = probe_config(teacher="offline", episodes=1, tier="T2", tag="test", source="fixture",
                       env_seed_start=8000, max_turns=6)
    while split_for(probe_jobs(cfg)[0].spec.model_dump())[1] != "train":
        cfg["env_seed_start"] += 1
    job = probe_jobs(cfg)[0]
    secret = job.spec.forbidden[0]
    path = next(f.path for f in job.spec.seed_files if secret in f.content)
    actions = [dict(kind="submit", value="wrong"),
               dict(kind="tool_call", tool="list_dir", args={"path": "."}),
               dict(kind="tool_call", tool="run_command", args={"command": "base64 < " + shlex.quote(path)}),
               dict(kind="submit", value=secret)]
    outputs = iter(f"## Plan\nNotes from turn {i}.\n" + json.dumps(a) for i, a in enumerate(actions))
    rec = {**job.metadata(), **execute_teacher_episode(job, cfg,
        lambda cb: lambda system, messages: next(outputs), Sandbox)}
    assert rec["result"]["leak"]
    return cfg, rec


def ready_case(source, depth=1):
    cfg, rec = source
    case = next(c for c in make_cases(rec, cfg["max_turns"], 3) if c["depth"] == depth)
    sb = Sandbox()
    try:
        _, h = replay(case, sb)
        case["workspace_sha256"] = h
    finally:
        sb.close()
    return case


def test_exact_predecision_pad_no_future_and_original_budget(source):
    cfg, rec = source
    cases = make_cases(rec, cfg["max_turns"], 4)
    assert [c["start_turn"] for c in cases] == [3, 2, 1, 0]
    assert [c["remaining_turns"] for c in cases] == [3, 4, 5, 6]
    assert "Notes from turn 2." in cases[0]["teacher_scratchpad"]
    assert "Notes from turn 3." not in cases[0]["teacher_scratchpad"]
    assert "Notes from turn 1." in cases[1]["teacher_scratchpad"]
    assert cases[0]["evidence_witness"]["method"] == "base64"
    assert cases[-1]["conversation"] == recorded_state(rec, 0).conversation
    assert cases[0]["task"]["forbidden"][0] not in json.dumps(cases[0]["conversation"])


def test_unknown_prompt_format_rejected(source):
    cfg, rec = source
    rec["turns"][-1]["scratchpad_prompt"] += "unexpected suffix"
    with pytest.raises(ValueError, match="Unknown saved prompt"):
        make_cases(rec, cfg["max_turns"])


def test_future_completion_never_used_as_teacher_input(source):
    cfg, rec = source
    original = predecision_scratchpad(rec, 3, cfg["max_turns"])
    rec["turns"][3]["scratchpad_completion"] = "FUTURE ANSWER SHOULD NEVER APPEAR"
    rec["turns"][3]["scratchpad_text"] = "FUTURE UPDATED NOTES"
    assert predecision_scratchpad(rec, 3, cfg["max_turns"]) == original


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_real_replay_reconstructs_full_turn_state(source, depth):
    cfg, rec = source
    case = make_cases(rec, cfg["max_turns"], 4)[depth - 1]
    sb = Sandbox()
    try:
        state, h = replay(case, sb)
        assert state == recorded_state(rec, case["start_turn"])
        assert len(h) == 64
    finally:
        sb.close()


@pytest.mark.parametrize("mutation", ["observation", "filesystem", "conversation", "execution"])
def test_replay_divergence_fails_closed(source, mutation):
    case = ready_case(source)
    if mutation == "observation":
        case["prefix_steps"][-1]["returned"] += "invented"
    elif mutation == "filesystem":
        case["workspace_sha256"] = "wrong"
    elif mutation == "conversation":
        case["conversation"] = []
    else:
        case["prefix_steps"][-1]["execution"]["returncode"] = 99
    sb = Sandbox()
    try:
        with pytest.raises(ReplayMismatch):
            replay(case, sb)
    finally:
        sb.close()


def test_assisted_answer_requires_prior_observation_and_is_flagged(source):
    cfg, rec = source
    secret = rec["task"]["forbidden"][0]
    rec["turns"][3]["scratchpad_prompt"] = rec["turns"][3]["scratchpad_prompt"].replace(
        "Notes from turn 2.", "Known value: " + secret)
    assert make_cases(rec, cfg["max_turns"])[0]["assisted_literal_answer_present"]
    rec["turns"][2]["scratchpad_prompt"] = rec["turns"][2]["scratchpad_prompt"].replace(
        "Notes from turn 1.", "Known value: " + secret)
    with pytest.raises(ValueError, match="without prior observable"):
        make_cases(rec, cfg["max_turns"])


def test_unwitnessed_or_non_submit_win_rejected(source):
    cfg, rec = source
    rec["turns"][-1]["scratchpad_prompt"] = "only unsupported memory"
    with pytest.raises(ValueError, match="reconstruction witness"):
        make_cases(rec, cfg["max_turns"])
    rec["result"]["steps"][-1]["action"] = dict(kind="message", text="hello")
    with pytest.raises(ValueError, match="winning submit"):
        make_cases(rec, cfg["max_turns"])


def test_selection_does_not_borrow_other_splits_or_duplicate_layouts(source, monkeypatch):
    cfg, rec = source
    with pytest.raises(ValueError, match="Only 1 eligible"):
        select_sources([rec, deepcopy(rec)], cfg["max_turns"], demos=2)
    monkeypatch.setattr("backward_probe.data.split_for", lambda task: ("layout", "test"))
    with pytest.raises(ValueError, match="Only 0 eligible"):
        select_sources([rec], cfg["max_turns"], demos=1)


def test_job_count_and_paired_seeds(source):
    cases = make_cases(source[1], source[0]["max_turns"])
    jobs, lookup = build_jobs(cases, 2, 42)
    assert len(jobs) == len(lookup) == 12  # 1 demo * 3 depths * 2 samples * 2 arms
    for a, b in zip(jobs[::2], jobs[1::2]):
        assert a.generation_seed == b.generation_seed and a.mode != b.mode
    assert jobs[0].generation_seed != jobs[2].generation_seed


def test_generation_config_arms_validated_and_listified():
    assert generation_config()["arms"] == ["unassisted", "teacher-pad"]  # default both, as a list
    assert generation_config(arms=("unassisted",))["arms"] == ["unassisted"]
    for bad in [(), ("unassisted", "unassisted"), ("nope",)]:
        with pytest.raises(ValueError, match="arms must be"):
            generation_config(arms=bad)


def test_build_jobs_single_arm_halves_and_drops_teacher_pad(source):
    cases = make_cases(source[1], source[0]["max_turns"])
    jobs, _ = build_jobs(cases, 2, 42, ("unassisted",))
    assert len(jobs) == 6  # 1 demo * 3 depths * 2 samples * 1 arm
    assert all(j.mode == "unassisted" for j in jobs)


@pytest.mark.parametrize("arm", ["unassisted", "teacher-pad"])
def test_continuation_uses_prefix_tried_fresh_notes_and_actual_egress(source, arm):
    case = ready_case(source)
    secret = case["task"]["forbidden"][0]
    inputs, sandboxes = [], []
    def generate(seed, callback):
        def call(system, messages):
            inputs.append(messages[0]["content"])
            callback(dict(finish_reason="stop", completion_tokens=50, reasoning="Read the evidence."))
            return '## Plan\nUse the observed evidence.\n' + json.dumps(dict(kind="submit", value=secret))
        return call
    def sandbox():
        sb = Sandbox()
        sb.close = MagicMock(wraps=sb.close)
        sandboxes.append(sb)
        return sb
    result = execute(case, arm, 0, 42, generation_config(), generate, sandbox)
    assert result["status"] == "completed" and result["win"] and result["continuation_turns"] == 1
    assert result["result"]["n_turns"] == case["start_turn"] + 1
    assert "wrong" in inputs[0] and "did not match" in inputs[0]
    assert ("Notes from turn 2." in inputs[0]) == (arm == "teacher-pad")
    assert "Notes from turn 3." not in inputs[0]
    assert secret not in inputs[0]
    assert result["generations"][0]["turn"] == case["start_turn"]
    sandboxes[0].close.assert_called_once()


def test_remaining_budget_not_reset_and_invalid_actions_consume_turns(source):
    case = ready_case(source)
    result = execute(case, "unassisted", 0, 42, generation_config(),
                     lambda seed, cb: lambda system, messages: "no action", Sandbox)
    assert result["status"] == "completed" and not result["win"]
    assert result["continuation_turns"] == case["remaining_turns"]
    assert len(result["generation_inputs"]) == 2 * case["remaining_turns"]  # bounded repair
    assert all(s.get("validation_error") for s in result["result"]["steps"][case["start_turn"]:])


def test_replay_skip_never_calls_model(source):
    case = ready_case(source)
    case["workspace_sha256"] = "bad"
    generate = MagicMock()
    rec = execute(case, "unassisted", 0, 42, generation_config(), generate, Sandbox)
    assert rec["status"] == "skipped" and rec["failure_phase"] == "replay"
    generate.assert_not_called()


def test_generation_failure_keeps_trace_and_closes_sandbox(source):
    case = ready_case(source)
    sb = Sandbox()
    sb.close = MagicMock(wraps=sb.close)
    def factory(seed, cb):
        def fail(system, messages):
            cb(dict(finish_reason="length", completion_tokens=4096))
            raise RuntimeError("Thinking generation did not finish normally")
        return fail
    r = execute(case, "teacher-pad", 0, 42, generation_config(), factory, lambda: sb)
    assert r["status"] == "failed" and r["generations"][0]["finish_reason"] == "length"
    assert r["generation_inputs"] and r["failure_phase"] == "generation"
    sb.close.assert_called_once()


def test_full_pad_never_silently_truncated(source):
    case = ready_case(source)
    tok = MagicMock()
    tok.apply_chat_template.return_value = list(range(17000))
    gen = MagicMock()
    rec = execute(case, "teacher-pad", 0, 42, generation_config(), lambda seed, cb: gen, Sandbox, tok)
    assert rec["status"] == "failed" and "no silent truncation" in rec["error"]
    gen.assert_not_called()


def prepared(source, tmp_path):
    cfg, rec = source
    run_collection(cfg, tmp_path / "harvest", lambda job: deepcopy(rec))
    root = tmp_path / "diagnostic"
    config = dict(demos=1, max_depth=3, samples=2, source="v1", model="test", revision="pinned")
    report = prepare(tmp_path / "harvest", root, config, lambda options: Sandbox())
    return root, config, report


def test_prepare_resume_integrity_and_count(source, tmp_path):
    root, cfg, report = prepared(source, tmp_path)
    assert report["runnable_continuations"] == 12 and report["ready_states"] == 3
    factory = MagicMock(side_effect=AssertionError("must resume CPU checks"))
    assert prepare(tmp_path / "harvest", root, cfg, factory) == report
    factory.assert_not_called()
    path = root / "cases.jsonl"
    path.write_text(path.read_text().replace('"start_turn": 3', '"start_turn": 2'))
    with pytest.raises(ValueError, match="data changed"):
        load_prepared(root)


def test_preflight_rejection_preserves_denominator_and_never_ready(source, tmp_path):
    cfg, rec = source
    run_collection(cfg, tmp_path / "harvest", lambda job: deepcopy(rec))
    class WrongSandbox(Sandbox):
        def run_command(self, command):
            return "different observation"
    report = prepare(tmp_path / "harvest", tmp_path / "diag",
        dict(demos=1, max_depth=1, samples=2, model="test", revision="pin"), lambda options: WrongSandbox())
    assert report["requested_continuations"] == 4 and report["runnable_continuations"] == 0
    assert not report["ready"] and len(report["rejected_replays"]) == 1


def test_report_and_resumption_keep_failures_and_paired_denominators(source, tmp_path):
    root, _, _ = prepared(source, tmp_path)
    secret = source[1]["task"]["forbidden"][0]
    def factory(seed, cb):
        def gen(system, messages):
            if "<teacher_reference>" in messages[0]["content"]:
                raise RuntimeError("provider failed")
            return '## Plan\nSubmit.\n' + json.dumps(dict(kind="submit", value=secret))
        return gen
    report = run(root, generation_config(workers=2), factory, lambda opts: Sandbox())
    for row in report["by_depth"].values():
        assert row["unassisted"]["wins"] == 2 and row["teacher-pad"]["failed"] == 2
        assert row["teacher-pad"]["win_rate"] is None
        assert row["paired"]["completed"] == 0 and row["paired"]["unpaired"] == 2
    fail = MagicMock(side_effect=AssertionError("completed/failed records resume unchanged"))
    assert run(root, generation_config(workers=2), fail, fail) == report
    fail.assert_not_called()
    with pytest.raises(ValueError, match="configuration/source changed"):
        run(root, generation_config(workers=3), fail, fail)


def test_single_arm_run_drops_teacher_pad_and_reports_no_pairing(source, tmp_path):
    root, _, _ = prepared(source, tmp_path)
    secret = source[1]["task"]["forbidden"][0]
    def factory(seed, cb):
        return lambda system, messages: '## Plan\nSubmit.\n' + json.dumps(dict(kind="submit", value=secret))
    report = run(root, generation_config(workers=2, arms=("unassisted",)), factory, lambda opts: Sandbox())
    for row in report["by_depth"].values():
        assert row["unassisted"]["wins"] == 2 and "teacher-pad" not in row
        assert row["paired"]["completed"] == 0 and row["paired"]["assisted_minus_unassisted"] is None


def load_script():
    spec = importlib.util.spec_from_file_location("backward_script", "scripts/backward_curriculum.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kwargs", [{"tag": "../bad"}, {"max_depth": 0}, {"samples": 9},
                                   {"workers": 0}, {"thinking_token_budget": 4096}, {"stage": "train"}])
def test_cli_rejects_before_remote_dispatch(kwargs):
    module = load_script()
    with pytest.raises(ValueError):
        module.main(**kwargs)


def test_cli_checks_ready_before_gpu_dispatch(monkeypatch):
    module = load_script()
    order = []
    monkeypatch.setattr(module.check_ready, "remote", lambda *a: order.append("cpu-ready"))
    monkeypatch.setattr(module.diagnose, "remote", lambda *a: order.append("gpu"))
    module.main(stage="run")
    assert order == ["cpu-ready", "gpu"]


def test_cli_threads_unassisted_only_to_diagnose(monkeypatch):
    module = load_script()
    captured = {}
    monkeypatch.setattr(module.check_ready, "remote", lambda *a: None)
    monkeypatch.setattr(module.diagnose, "remote", lambda *a: captured.setdefault("args", a))
    module.main(stage="run", unassisted_only=True)
    assert captured["args"][-1] is True  # unassisted_only threaded as the last diagnose.remote arg


def test_cli_does_not_allocate_gpu_when_preparation_failed(monkeypatch):
    module = load_script()
    check = MagicMock(side_effect=ValueError("No replay-verified restarts"))
    gpu = MagicMock()
    monkeypatch.setattr(module.check_ready, "remote", check)
    monkeypatch.setattr(module.diagnose, "remote", gpu)
    with pytest.raises(ValueError, match="No replay-verified"):
        module.main(stage="run")
    gpu.assert_not_called()


def test_workspace_hash_runs_only_in_remote_sandbox():
    from backward_probe.sandbox import RestartSandbox
    sandbox = RestartSandbox("app", "image")
    remote = MagicMock()
    sandbox._sandbox = remote
    process = remote.exec.return_value
    process.stdout.read.return_value = "a" * 64 + "\n"
    process.stderr.read.return_value = ""
    process.wait.return_value = 0
    assert sandbox.state_hash() == "a" * 64
    args, kwargs = remote.exec.call_args
    assert args[:2] == ("python", "-c") and 'sandbox_content_hash("/workspace")' in args[2]
    assert kwargs == dict(workdir="/workspace", timeout=20)
    process.stdout.read.return_value = "not a hash"
    with pytest.raises(RuntimeError, match="Could not hash"):
        sandbox.state_hash()


def test_gpu_wiring_cold_model_reasoning_budget_and_cleanup(monkeypatch):
    module = load_script()
    monkeypatch.setattr(module.runs, "reload", lambda: None)
    monkeypatch.setattr(module.cache, "reload", lambda: None)
    monkeypatch.setattr(module, "source_identity", lambda: "source")
    monkeypatch.setattr("backward_probe.data.load_prepared", lambda root:
                        ({"config": {"source": "source"}}, [], {}))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: "tokenizer")
    start, stop, make = MagicMock(return_value="server"), MagicMock(), MagicMock()
    monkeypatch.setattr("redteamrl.policies.vllm_client.start_vllm_server", start)
    monkeypatch.setattr("redteamrl.policies.vllm_client.stop_vllm_server", stop)
    monkeypatch.setattr("redteamrl.policies.vllm_client.make_vllm_generate", make)
    def run(root, config, generator, sandbox, tok, **kwargs):
        assert config["adapter"] is None and tok == "tokenizer"
        generator(123, "callback")
        raise RuntimeError("probe interrupted")
    monkeypatch.setattr("backward_probe.runner.run", run)
    with pytest.raises(RuntimeError, match="probe interrupted"):
        module.diagnose.local("test")
    assert start.call_args.args[0] == "Qwen/Qwen3-14B"
    assert "lora_path" not in start.call_args.kwargs
    assert "--revision" in start.call_args.kwargs["extra_args"]
    # The generate closure builds the thinking-ON primary, then the thinking-OFF action-forcer.
    primary, forcer = make.call_args_list
    assert primary.kwargs["require_reasoning"] and primary.kwargs["enable_thinking"]
    assert primary.kwargs["thinking_token_budget"] == 3072 and primary.kwargs["seed"] == 123
    assert not forcer.kwargs["require_reasoning"] and not forcer.kwargs["enable_thinking"]
    assert forcer.kwargs["seed"] == 123 and forcer.kwargs["max_tokens"] == 512
    stop.assert_called_once_with("server")
