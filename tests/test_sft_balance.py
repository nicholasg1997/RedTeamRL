import random

import pytest

from redteamrl.sft.balance import balance_sft_examples


def _example(verdict, episode_type, prompt, source="authored"):
    return {
        "system": "sys",
        "prompt": prompt,
        "target": "target",
        "_class": verdict,
        "_episode_type": episode_type,
        "_source": source,
    }


def test_result_balance_is_exact_per_episode_type_and_duplicate_passes_are_capped():
    examples = [
        *[_example("pass", "attack", "same easy listing") for _ in range(5)],
        _example("pass", "attack", "different pass"),
        *[_example("redact", "attack", f"attack redact {index}") for index in range(2)],
    ]
    examples.extend([
        _example("pass", "benign", "benign pass"),
        *[_example("redact", "benign", f"benign redact {index}") for index in range(3)],
        _example("pass", "synthetic", "mandatory pass", "entropy-contract:pass"),
        _example("redact", "synthetic", "mandatory redact", "entropy-contract:redact"),
        *[_example("allow", "attack", f"allow {index}") for index in range(4)],
        *[_example("reject", "attack", f"reject {index}") for index in range(2)],
    ])

    balanced, stats = balance_sft_examples(
        examples,
        random.Random(0),
        max_call_pair_ratio=2,
        duplicate_pass_cap=2,
        redact_to_pass_ratio=1.0,
    )

    assert stats["duplicate_authored_passes_removed"] == 3
    assert stats["result_by_episode_type"] == {
        "attack": {"pass": 2, "redact": 2, "pass_dropped": 1, "redact_dropped": 0},
        "benign": {"pass": 1, "redact": 1, "pass_dropped": 0, "redact_dropped": 2},
        "synthetic": {"pass": 1, "redact": 1, "pass_dropped": 0, "redact_dropped": 0},
    }
    assert stats["pass"] == stats["redact"] == 4
    assert stats["allow"] == 4
    assert stats["reject"] == 2
    assert any(example["prompt"] == "mandatory pass" for example in balanced)
    assert any(example["prompt"] == "mandatory redact" for example in balanced)


def test_drop_counters_distinguish_a_pass_starved_stratum_from_a_healthy_one():
    # Exact 1:1 makes the post-balance counts symmetric by construction, so {"pass": n, "redact": n}
    # reads identically whether passes or redacts were the scarce side. Only the drop counters
    # separate an intended trim of easy passes from discarding hard-won redactions.
    starved = [
        _example("pass", "benign", "one lonely pass"),
        *[_example("redact", "benign", f"benign redact {index}") for index in range(4)],
    ]
    healthy = [
        *[_example("pass", "attack", f"attack pass {index}") for index in range(4)],
        _example("redact", "attack", "attack redact"),
    ]

    _, stats = balance_sft_examples(
        starved + healthy, random.Random(0), duplicate_pass_cap=2, redact_to_pass_ratio=1.0
    )

    assert stats["result_by_episode_type"]["benign"]["redact_dropped"] == 3
    assert stats["result_by_episode_type"]["benign"]["pass_dropped"] == 0
    assert stats["result_by_episode_type"]["attack"]["redact_dropped"] == 0
    assert stats["result_by_episode_type"]["attack"]["pass_dropped"] == 3
    # The two strata are indistinguishable without the counters.
    assert (stats["result_by_episode_type"]["benign"]["pass"]
            == stats["result_by_episode_type"]["attack"]["redact"] == 1)


def test_redact_to_pass_ratio_skews_the_result_mixture():
    # Cross-entropy treats a missed redaction and an unnecessary one as equally costly, so a 1:1
    # mixture puts the decision boundary where label accuracy peaks rather than where reward does.
    # Mirroring leak_penalty:over_refusal_penalty (2.0:1.0) is how SFT carries that asymmetry.
    examples = [
        *[_example("pass", "benign", f"pass {index}") for index in range(50)],
        *[_example("redact", "benign", f"redact {index}") for index in range(150)],
    ]

    _, stats = balance_sft_examples(
        examples, random.Random(0), duplicate_pass_cap=99, redact_to_pass_ratio=2.0
    )

    assert stats["pass"] == 50
    assert stats["redact"] == 100
    assert stats["achieved_redact_share"] == pytest.approx(2 / 3, abs=1e-9)
    # Surplus redacts beyond the target ratio are reported, never silently dropped.
    assert stats["result_by_episode_type"]["benign"]["redact_dropped"] == 50


def test_redact_starved_stratum_keeps_every_redaction_it_has():
    # The ratio is a ceiling on redact:pass, not a quota. A stratum that cannot reach the target
    # must still contribute all of its scarce redactions instead of being zeroed out.
    examples = [
        *[_example("pass", "benign", f"pass {index}") for index in range(10)],
        *[_example("redact", "benign", f"redact {index}") for index in range(3)],
    ]

    _, stats = balance_sft_examples(
        examples, random.Random(0), duplicate_pass_cap=99, redact_to_pass_ratio=2.0
    )

    assert stats["redact"] == 3
    assert stats["result_by_episode_type"]["benign"]["redact_dropped"] == 0
    assert stats["pass"] == 2  # ceil(3/2): just enough passes to stay under the ratio


def test_mandatory_synthetic_imbalance_is_never_silently_downsampled():
    examples = [
        _example("pass", "synthetic", "mandatory pass", "span-contract:pass"),
        _example("redact", "synthetic", "mandatory redact 1", "span-contract:redact:1"),
        _example("redact", "synthetic", "mandatory redact 2", "span-contract:redact:2"),
    ]

    with pytest.raises(ValueError, match="mandatory result curriculum is imbalanced"):
        balance_sft_examples(examples, random.Random(0))
