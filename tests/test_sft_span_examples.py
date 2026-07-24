import json

from redteamrl.sft.span_examples import build_span_counterexamples


def _decision(example):
    return json.loads(example["target"].splitlines()[-1])


def test_span_counterexamples_are_balanced_and_remove_values_not_labels():
    examples = build_span_counterexamples(variants_per_format=2)
    redacts = [example for example in examples if example["_class"] == "redact"]
    passes = [example for example in examples if example["_class"] == "pass"]

    assert len(redacts) == 12
    assert len(passes) == 14
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
    ]

    assert len(filename_examples) == 1
    assert filename_examples[0]["_class"] == "pass"
    assert "credentials" in filename_examples[0]["prompt"]
