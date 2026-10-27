from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import pytest

from overfit_probe.data import choose_cases, load_cases
from overfit_probe.runner import (DEFAULT_CONFIG, action_encoded, baseline_ceiling, decision_report, grade,
                                  messages_for, probe_cases, served_probes, validate_config)
from redteamrl.scratchpad.experiment import atomic_json, ensure_manifest
from teacher_sft.data import digest, write_jsonl
from teacher_sft.encoding import encode_example


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"]
        return "".join(f"<{m['role']}>\n{m['content']}\x03" for m in messages) + (
            "<assistant>\n" if kwargs["add_generation_prompt"] else "")

    def __call__(self, text, **kwargs):
        result = {"input_ids": [ord(c) for c in text]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def fixtures(n_pairs=4):
    examples, originals, tasks = [], [], []
    for i in range(n_pairs):
        episode, group, secret = f"ep-{i}", f"layout-{i}", f"canary-{i}"
        tasks.append(dict(episode=episode, group=group, split="train", task={"forbidden": [secret]}))
        for j in (0, 1):
            action = {"kind": "tool_call", "tool": "read_file", "args": {"path": f"evidence-{i}.txt"}} if j == 0 else {"kind": "submit", "value": secret}
            proof = {"method": "base64", "observation_turn": 0} if j else None
            common = dict(id=f"{episode}:{j}:natural", episode=episode, group=group, split="train",
                          kind="natural", system="Original system", prompt=f"Original observable state {i}-{j}",
                          action=action, tools=["read_file"], witness=proof)
            original = {**common, "original_outcome": "leak"}
            originals.append(original)
            examples.append({**common, "candidate_sha256": digest(original), "reasoning": "Use the evidence.",
                             "final": "## Plan\nCheck the evidence.\n" + json.dumps(action)})
    return examples, originals, tasks


def cases_fixture():
    cases, _ = choose_cases(*fixtures(), Tokenizer())
    return cases


def selected(root):
    cases = cases_fixture()
    ensure_manifest(root, {"cases_sha256": digest(cases), "model": "test", "revision": "pinned",
                            "max_length": 8192, "max_new_tokens": 3072})
    write_jsonl(root / "cases.jsonl", cases)
    return cases


def response(case, exact=True, **changes):
    action = case["action"] if exact else {"kind": "message", "text": "another action"}
    return dict(reasoning="Brief reasoning", content="## Plan\nAct.\n" + json.dumps(action), finish_reason="stop") | changes


def test_selection_keeps_real_pairs_and_is_order_invariant():
    rows, originals, tasks = fixtures(6)
    a, _ = choose_cases(rows, originals, tasks, Tokenizer())
    b, _ = choose_cases(list(reversed(rows)), originals, tasks, Tokenizer())
    assert a == b and len(a) == 8
    assert len({r["group"] for r in a}) == 4
    for i in range(0, 8, 2):
        assert a[i]["episode"] == a[i + 1]["episode"]
        assert [r["probe_role"] for r in a[i:i + 2]] == ["evidence", "submit"]


def test_selection_never_borrows_dev_or_teacher_hints():
    rows, originals, tasks = fixtures(4)
    rows[0]["reasoning"] = "The selected action is supplied."
    rows[2]["split"] = "dev"
    with pytest.raises(ValueError, match="Do not borrow dev/test"):
        choose_cases(rows, originals, tasks, Tokenizer())


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row.update(candidate_sha256="changed"), "match its prepared"),
    (lambda row: row.update(prompt="changed prompt"), "observable prompt"),
    (lambda row: row.update(group="another"), "provenance mismatch"),
    (lambda row: row.update(action={"kind": "message", "text": "wrong"}), "actual winning"),
])
def test_selection_rejects_changed_provenance(mutation, match):
    rows, originals, tasks = fixtures()
    mutation(rows[0])
    with pytest.raises(ValueError, match=match):
        choose_cases(rows, originals, tasks, Tokenizer())


def test_literal_answers_and_overlong_examples_not_silently_used():
    rows, originals, tasks = fixtures(4)
    for row, original in zip(rows, originals):
        if row["action"]["kind"] == "submit":
            original["prompt"] = row["prompt"] = row["action"]["value"]
            row["candidate_sha256"] = digest(original)
    with pytest.raises(ValueError, match="answer_already_literal"):
        choose_cases(rows, originals, tasks, Tokenizer())
    with pytest.raises(ValueError, match="encoding"):
        choose_cases(*fixtures(), Tokenizer(), max_length=10)


def test_action_loss_masks_reasoning_and_notes_and_preserves_ids():
    case = cases_fixture()[0]
    tokenizer = Tokenizer()
    ids, labels, weights = action_encoded(tokenizer, case, 8192)
    assert ids == encode_example(tokenizer, case)[0]
    assert json.loads("".join(chr(x) for x in labels if x != -100)) == case["action"]
    assert sum(weights) < sum(encode_example(tokenizer, case)[2])


def test_generation_input_excludes_all_reference_targets():
    case = cases_fixture()[0]
    assert messages_for(case) == [dict(role="system", content=case["system"]),
                                 dict(role="user", content=case["prompt"])]
    assert case["reasoning"] not in str(messages_for(case))
    assert case["final"] not in str(messages_for(case))


def test_exact_action_and_format_are_separate_and_truncation_not_success():
    case = cases_fixture()[0]
    assert grade(case, response(case))["exact_action"]
    no_think = grade(case, response(case, reasoning=""))
    assert no_think["exact_action"] and not no_think["format_ok"]
    assert not grade(case, response(case, finish_reason="length"))["exact_action"]
    assert not grade(case, response(case, exact=False))["exact_action"]


def test_probe_resume_and_coordinator_commits(tmp_path):
    cases = cases_fixture()
    commits = []
    coordinator = threading.get_ident()
    records = probe_cases(cases, tmp_path, response, identity="i", workers=4,
                          commit=lambda: commits.append(threading.get_ident()))
    assert set(commits) == {coordinator} and all(r["exact_action"] for r in records)
    assert probe_cases(cases, tmp_path, lambda c: pytest.fail("resume"), identity="i") == records
    with pytest.raises(ValueError, match="changed"):
        probe_cases(cases, tmp_path, response, identity="changed")


@pytest.mark.parametrize("variant,verdict", [
    ("learn", "PASS_SEEN_STATE_LEARNING"),
    ("baseline", "BASELINE_ALREADY_SOLVES"),
    ("no_update", "FAIL_NO_PARAMETER_CHANGE"),
    ("serve_fail", "INVESTIGATE_SERVING_DIFFERENCE"),
    ("no_learning", "NOT_DEMONSTRATED_AT_THIS_BUDGET"),
    ("length", "INCONCLUSIVE_GENERATION_ERRORS"),
])
def test_report_does_not_confuse_memorization_with_generalization(tmp_path, variant, verdict):
    cases = cases_fixture()
    for name in ("hf-base", "hf-tuned", "vllm-base", "vllm-tuned"):
        exact = variant == "baseline" or "tuned" in name
        if variant == "serve_fail" and name == "vllm-tuned" or variant == "no_learning":
            exact = False
        def generate(c):
            return response(c, exact=exact, finish_reason="length" if variant == "length" else "stop")
        probe_cases(cases, tmp_path / name, generate, identity="i")
    atomic_json(tmp_path / "fit.json", dict(parameters_changed=variant != "no_update",
        seen_loss_before={"all_tokens": 1.}, seen_loss_after={"all_tokens": .2}, step=40,
        presentations=160, target_tokens_seen=50000))
    report = decision_report(tmp_path)
    assert report["verdict"] == verdict and not report["promoted"]
    assert "no autonomous win" in report["purpose"]


def test_fresh_serving_process_uses_correct_alias_and_always_closes(tmp_path, monkeypatch):
    from redteamrl.policies import vllm_client as v
    cases = selected(tmp_path)
    aliases, stopped, starts = [], [], []
    def start(*args, **kwargs):
        starts.append(kwargs)
        return "process"
    def factory(url, alias, **kwargs):
        aliases.append(alias)
        assert kwargs["enable_thinking"] and kwargs["require_reasoning"]
        assert kwargs["temperature"] == 0. and "thinking_token_budget" not in kwargs
        def call(system, messages):
            case = next(c for c in cases if c["prompt"] == messages[0]["content"])
            r = response(case)
            kwargs["on_response"](dict(reasoning=r["reasoning"], final_content=r["content"], finish_reason="stop"))
        return call
    monkeypatch.setattr(v, "start_vllm_server", start)
    monkeypatch.setattr(v, "stop_vllm_server", stopped.append)
    monkeypatch.setattr(v, "make_vllm_generate", factory)
    fit = dict(config=DEFAULT_CONFIG | {"model": "test", "revision": "pinned"},
               adapter="saved-adapter", run_identity="i")
    served_probes(tmp_path, fit)
    assert set(aliases) == {"test", "overfit"} and len(aliases) == 16
    assert starts[0]["max_lora_rank"] == 16 and starts[0]["lora_path"] == "saved-adapter"
    assert stopped == ["process"]


def test_case_integrity_and_budget_bounds(tmp_path):
    cases = selected(tmp_path)
    cases[0]["prompt"] = "changed"
    write_jsonl(tmp_path / "cases.jsonl", cases)
    with pytest.raises(ValueError, match="changed"):
        load_cases(tmp_path)
    for changes in ({"max_steps": 0}, {"max_target_tokens": 300000}, {"rank": 32}):
        with pytest.raises(ValueError):
            validate_config(DEFAULT_CONFIG | changes)


def test_baseline_ceiling_can_skip_training_but_errors_cannot(tmp_path):
    cases = cases_fixture()
    rows = probe_cases(cases, tmp_path / "hf-base", response, identity="i")
    assert baseline_ceiling(rows)
    rows[0]["finish_reason"] = "length"
    assert not baseline_ceiling(rows)
    probe_cases(cases, tmp_path / "vllm-base", response, identity="i")
    atomic_json(tmp_path / "fit.json", dict(skipped_training=True, parameters_changed=False,
        seen_loss_before={"all_tokens": 1.}, seen_loss_after=None, step=0,
        presentations=0, target_tokens_seen=0))
    report = decision_report(tmp_path)
    assert report["verdict"] == "BASELINE_ALREADY_SOLVES" and report["step"] == 0


@pytest.mark.parametrize("eos", [True, False])
def test_hf_generation_uses_original_prompt_and_preserves_thinking(eos):
    import torch
    from overfit_probe.runner import hf_generate
    case = cases_fixture()[0]
    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            return self
    class HFTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            assert text == self.apply_chat_template(messages_for(case), enable_thinking=True, add_generation_prompt=True)
            return Batch(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones((1, 2)))
        def decode(self, ids, **kwargs):
            return "".join(chr(x) for x in ids if x != 3)
    class Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=[3])
        def eval(self):
            pass
        def generate(self, **kwargs):
            assert not kwargs["do_sample"] and kwargs["max_new_tokens"] == 3072
            assert not torch.is_grad_enabled()
            raw = "<think>\nBrief reason.\n</think>\n" + response(case)["content"]
            return torch.tensor([[1, 2] + [ord(c) for c in raw] + ([3] if eos else [])])
    result = hf_generate(Model(), HFTokenizer(), case, DEFAULT_CONFIG)
    assert result["reasoning"] == "Brief reason."
    assert result["finish_reason"] == ("stop" if eos else "length")
    assert grade(case, result)["exact_action"] == eos


def test_modal_cli_rejects_before_dispatch(monkeypatch):
    from scripts import scratchpad_overfit as script
    monkeypatch.setattr(script, "run", SimpleNamespace(remote=lambda *a: pytest.fail("dispatched")))
    for changes in ({"tag": "../escape"}, {"max_steps": 1000}, {"max_target_tokens": 0}):
        with pytest.raises(ValueError):
            script.main(stage="run", **changes)
