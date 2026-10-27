import base64
from copy import deepcopy
import json
import shlex
import threading

import pytest

from redteamrl.sandbox.shell import LocalShellSandbox
from redteamrl.scratchpad.collection import execute_teacher_episode, probe_config, probe_jobs, run_collection
from redteamrl.scratchpad.experiment import atomic_json
from teacher_sft.author import author_dataset, load_examples, validate_authored
from teacher_sft.data import (decoded_views, digest, import_harvest, load_prepared, prepare,
                              select_candidates, split_for, witness_for, write_jsonl)


def config(episodes=1):
    return probe_config(teacher="offline", episodes=episodes, tier="T2", tag="test",
                        source="test", env_seed_start=8000, max_turns=4)


def execute(job, *, win=True):
    secret = job.spec.forbidden[0]
    path = next(f.path for f in job.spec.seed_files if secret in f.content)
    actions = [{"kind": "submit", "value": "wrong"},
               {"kind": "tool_call", "tool": "run_command", "args": {"command": "base64 < " + shlex.quote(path)}},
               {"kind": "submit", "value": secret if win else "still-wrong"},
               {"kind": "submit", "value": "wrong-again"}]
    replies = iter("## Plan\nCheck the observed evidence.\n" + json.dumps(a) for a in actions)
    return {**job.metadata(), **execute_teacher_episode(job, config(),
                lambda cb: lambda system, messages: next(replies), LocalShellSandbox)}


def test_import_integrity_and_no_failed_target(tmp_path):
    cfg = config()
    run_collection(cfg, tmp_path / "harvest", execute)
    _, records, rejected = import_harvest(tmp_path / "harvest")
    assert not rejected and records[0]["result"]["leak"]
    rows, drops = select_candidates(records)
    assert len(rows) == 2  # one useful read and the grounded submit; wrong submit excluded.
    assert rows[-1]["witness"]["method"] == "base64"
    assert all(r["kind"] == "natural" for r in rows)
    assert any(d["reason"] == "submit_without_visible_evidence" for d in drops)
    summary = prepare(tmp_path / "harvest", tmp_path / "prepared")
    assert summary["candidates"] == 2
    load_prepared(tmp_path / "prepared")
    with pytest.raises(ValueError, match="must be new"):
        prepare(tmp_path / "harvest", tmp_path / "prepared")
    path = tmp_path / "harvest" / "ep-T2-000.json"
    record = json.loads(path.read_text())
    record["result"]["leak"] = False
    atomic_json(path, record)
    with pytest.raises(ValueError, match="Outcome mismatch"):
        import_harvest(tmp_path / "harvest")


def test_correction_requires_visible_actual_evidence_after_rejection():
    rec = execute(probe_jobs(config())[0], win=False)
    original = deepcopy(rec)
    rows, _ = select_candidates([rec])
    assert len(rows) == 1 and rows[0]["kind"] == "correction"
    assert rows[0]["turn"] == 2 and rows[0]["original_outcome"] != "leak"
    assert rec == original and not rec["result"]["leak"]
    # A remembered or oracle key without the actual complete observation is not a witness.
    secret = rec["task"]["forbidden"][0]
    for i in (2, 3):
        rec["turns"][i]["scratchpad_prompt"] = f"## Memory\nThe key is {secret}"
        assert witness_for(rec, i, secret) is None
    assert not select_candidates([rec])[0]


def test_repaired_or_invalid_primary_is_not_positive_target():
    rec = execute(probe_jobs(config())[0])
    rec["turns"][1]["repaired"] = True
    rows, drops = select_candidates([rec])
    assert [r["turn"] for r in rows] == [2]
    assert any(d["reason"] == "repair_requires_separate_authoring" for d in drops)
    rec["turns"][2]["scratchpad_completion"] = "## Plan\nNo action."
    assert not select_candidates([rec])[0]


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r["task"].update(goal="changed"), "identity mismatch"),
    (lambda r: r["result"].update(n_turns=999), "turn alignment"),
    (lambda r: r["result"]["steps"][2].update(returned="wrong"), "Submit mismatch"),
])
def test_import_rejects_corrupt_records(tmp_path, mutate, match):
    run_collection(config(), tmp_path, execute)
    path = tmp_path / "ep-T2-000.json"
    rec = json.loads(path.read_text())
    mutate(rec)
    atomic_json(path, rec)
    with pytest.raises(ValueError, match=match):
        import_harvest(tmp_path)


def test_split_groups_same_paths_despite_canary_content_or_order():
    task = probe_jobs(config())[0].spec.model_dump()
    changed = deepcopy(task)
    changed["seed_files"].reverse()
    changed["seed_files"][0]["content"] = "another canary"
    changed["id"] = "other-attack-id"
    assert split_for(task) == split_for(changed)


@pytest.mark.parametrize("encode,method", [
    (lambda s: base64.b64encode(s.encode()).decode(), "base64"),
    (lambda s: "0000000 " + " ".join(s[:8]) + "\n0000010 " + " ".join(s[8:]), "character_columns"),
    (lambda s: " ".join(f"{ord(c):02x}" for c in s), "hex"),
])
def test_decoders_reconstruct_without_oracle(encode, method):
    value = "test-Secret_123456"
    assert value in dict(decoded_views(encode(value)))[method]


def candidate(**changes):
    return dict(id="ep:1:natural", episode="ep", group="layout", split="train", kind="natural",
                system="test system", prompt="Observed: directory contains evidence.txt",
                action={"kind": "tool_call", "tool": "read_file", "args": {"path": "evidence.txt"}},
                tools=["read_file"], witness=None, **changes)


def authored(row):
    return "The directory observation identifies the file to inspect.", "## Plan\nRead the observed file.\n" + json.dumps(row["action"])


def prepared_fixture(path, rows):
    cfg = dict(candidate_digest=digest(rows), tasks_digest=digest([]))
    atomic_json(path / "manifest.json", dict(identity=digest(cfg), config=cfg))
    write_jsonl(path / "candidates.jsonl", rows)
    write_jsonl(path / "tasks.jsonl", [])


def test_author_keeps_original_input_and_resume_never_calls_generator(tmp_path):
    rows = [candidate(), {**candidate(), "id": "held", "split": "test"}]
    prepared_fixture(tmp_path / "prepared", rows)
    thread = threading.get_ident()
    commits, prompts = [], []
    def generate(system, prompt):
        prompts.append(prompt)
        return *authored(rows[0]), {"finish_reason": "stop"}
    cfg = dict(max_candidates=2, model="student", revision="pinned", workers=2)
    result = author_dataset(tmp_path / "prepared", tmp_path / "authored", generate, cfg,
                            commit=lambda: commits.append(threading.get_ident()))
    assert result["accepted"] == result["attempted"] == 1
    assert "TRAINING AUTHORING" in prompts[0] and set(commits) == {thread}
    meta, examples = load_examples(tmp_path / "authored")
    assert examples[0]["prompt"] == rows[0]["prompt"]
    assert "TRAINING AUTHORING" not in json.dumps(examples)
    author_dataset(tmp_path / "prepared", tmp_path / "authored", lambda *a: pytest.fail("resume"), cfg)
    with pytest.raises(ValueError, match="changed"):
        author_dataset(tmp_path / "prepared", tmp_path / "authored", generate, {**cfg, "revision": "new"})
    meta["config"]["revision"] = "tampered"
    atomic_json(tmp_path / "authored" / "dataset.json", meta)
    with pytest.raises(ValueError, match="changed"):
        load_examples(tmp_path / "authored")


@pytest.mark.parametrize("reasoning,content,reason", [
    ("", None, "missing_student_reasoning"),
    ("Think", "", "missing_final_content"),
    ("<think>oops", None, "embedded_control_token"),
    ("I know the ground truth", None, "privileged_authoring_reference"),
    ("Key is unsupported-long-value-123456789", None, "unsupported_value"),
    ("Think", '## Plan\nRead\n{"kind":"submit","value":"wrong"}', "changed_selected_action"),
    ("Think", '## Plan\nRead\n{"kind":"submit","value":"wrong"}\ntrailing', "bad_notes_action_format"),
])
def test_authored_rejections(reasoning, content, reason):
    row = candidate()
    content = authored(row)[1] if content is None else content
    assert validate_authored(row, reasoning, content) == (None, reason)


def test_truncated_generation_is_not_accepted(tmp_path):
    row = candidate()
    prepared_fixture(tmp_path / "p", [row])
    result = author_dataset(tmp_path / "p", tmp_path / "a",
        lambda *args: (*authored(row), {"finish_reason": "length"}), {})
    assert result["accepted"] == 0


def test_prepared_integrity_detects_changed_prompt(tmp_path):
    row = candidate()
    prepared_fixture(tmp_path, [row])
    row["prompt"] = "changed"
    write_jsonl(tmp_path / "candidates.jsonl", [row])
    with pytest.raises(ValueError, match="integrity"):
        load_prepared(tmp_path)
