import math

from redteamrl.train.diversity import group_diversity


def _ex(task_id, episode_id, reward, verdicts):
    return {"task_id": task_id, "episode_id": episode_id, "reward": reward, "verdicts": verdicts}


def test_identical_verdicts_across_a_group_is_a_dead_group():
    # reject_recall=0% is headroom, not gradient. If every sampled defender says "allow", every
    # reward is identical, every advantage is zero, and GRPO learns nothing from the group.
    group = [_ex("t", i, -2.0, ["allow", "allow"]) for i in range(6)]
    report = group_diversity(group)
    assert report["mixed_verdict_group_rate"] == 0.0
    assert report["mixed_reward_group_rate"] == 0.0
    assert report["verdict_entropy"] == 0.0
    assert report["dead_group_rate"] == 1.0


def test_mixed_verdicts_register_as_live():
    group = [_ex("t", i, -2.0 if i % 2 else 1.0, ["allow"] if i % 2 else ["reject"])
             for i in range(6)]
    report = group_diversity(group)
    assert report["mixed_verdict_group_rate"] == 1.0
    assert report["mixed_reward_group_rate"] == 1.0
    assert report["dead_group_rate"] == 0.0
    assert report["verdict_entropy"] == 1.0   # even 50/50 split over two verdicts


def test_verdict_variation_without_reward_variation_is_still_a_dead_group():
    # The distinction that matters: differing verdicts that all lead to the same outcome give
    # GRPO no signal either. Diversity is necessary, not sufficient.
    group = [_ex("t", i, -2.0, ["allow"] if i % 2 else ["reject"]) for i in range(6)]
    report = group_diversity(group)
    assert report["mixed_verdict_group_rate"] == 1.0
    assert report["mixed_reward_group_rate"] == 0.0
    assert report["dead_group_rate"] == 1.0


def test_constant_permissive_call_and_result_policy_is_not_fake_diversity():
    verdicts = [
        {"decision_type": "call", "verdict": "allow"},
        {"decision_type": "result", "verdict": "pass"},
    ]
    group = [_ex("t", i, -2.0, verdicts) for i in range(4)]

    report = group_diversity(group)

    assert report["mixed_verdict_group_rate"] == 0.0
    assert report["verdict_entropy"] == 0.0
    assert report["mixed_verdict_group_rate_by_type"] == {"call": 0.0, "result": 0.0}


def test_episode_verdict_trace_is_counted_once_when_episode_has_multiple_examples():
    verdicts = [{"decision_type": "call", "verdict": "allow"}]
    group = [
        _ex("t", 0, -2.0, verdicts),
        _ex("t", 0, -2.0, verdicts),
        _ex("t", 1, 1.0, [{"decision_type": "call", "verdict": "reject"}]),
    ]

    report = group_diversity(group)

    assert report["verdict_counts_by_type"]["call"] == {"allow": 1, "reject": 1}
    assert report["mixed_verdict_group_rate_by_type"]["call"] == 1.0
    assert report["dead_group_rate"] == 0.0


def test_entropy_is_normalized_and_per_group_rates_average_across_tasks():
    group = [
        *[_ex("a", i, 1.0 if i else -2.0, ["allow"] if i else ["reject"]) for i in range(4)],
        *[_ex("b", 10 + i, -2.0, ["allow"]) for i in range(4)],
    ]
    report = group_diversity(group)
    assert report["mixed_verdict_group_rate"] == 0.5
    assert report["dead_group_rate"] == 0.5
    # Task a is 3 allow / 1 reject -> entropy below 1; task b contributes 0.
    expected_a = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert math.isclose(report["verdict_entropy"], expected_a / 2, rel_tol=1e-9)


def test_empty_input_is_safe():
    report = group_diversity([])
    assert report["mixed_verdict_group_rate"] == 0.0
    assert report["dead_group_rate"] == 0.0
