import json

from redteamrl.sft.span_examples import build_span_counterexamples


def _decision(example):
    return json.loads(example["target"].splitlines()[-1])


def test_span_counterexamples_are_balanced_and_remove_values_not_labels():
    examples = build_span_counterexamples(variants_per_format=2)
    redacts = [example for example in examples if example["_class"] == "redact"]
    passes = [example for example in examples if example["_class"] == "pass"]

    assert len(redacts) == 22
    assert len(passes) == 22
    for example in redacts:
        decision = _decision(example)
        assert len(decision["remove"]) == 1
        removed = decision["remove"][0]
        assert removed in example["prompt"]
        assert len(removed) == 32
        assert not removed.endswith("TOKEN")
    for example in passes:
        assert _decision(example)["verdict"] == "pass"


def test_filename_credentials_is_a_pass_counterexample():
    examples = build_span_counterexamples(variants_per_format=1)
    filename_examples = [
        example for example in examples
        if example["_source"].startswith("span-contract:filename")
        and example["_class"] == "pass"
    ]

    assert len(filename_examples) == 1
    assert filename_examples[0]["_class"] == "pass"
    assert "credentials" in filename_examples[0]["prompt"]


def test_high_entropy_examples_pair_clean_pass_with_selective_redaction():
    examples = build_span_counterexamples(variants_per_format=1)
    entropy = [
        example for example in examples
        if example["_source"].startswith("entropy-contract:")
    ]

    assert len(entropy) == 8
    assert sum(example["_class"] == "pass" for example in entropy) == 4
    assert sum(example["_class"] == "redact" for example in entropy) == 4
    for example in entropy:
        assert "GIT_SHA" in example["prompt"] or "git_sha" in example["prompt"]
        assert "build_id" in example["prompt"].lower()
        assert "request_id" in example["prompt"].lower()
        assert "checksum" in example["prompt"].lower() or "sha256" in example["prompt"].lower()
        decision = _decision(example)
        if example["_class"] == "pass":
            assert decision["remove"] == []
        else:
            removed = decision["remove"]
            assert len(removed) == 1
            assert removed[0] in example["prompt"]
            assert "git_sha" not in removed[0].lower()
