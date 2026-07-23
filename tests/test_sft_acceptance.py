from redteamrl.sft.acceptance import evaluate_acceptance


def _report(attack, leak, completion, over_refusal):
    return {
        "overall_attack": {"attack_success_rate": attack},
        "overall_benign": {
            "benign_leak_rate": leak,
            "completion_rate": completion,
            "over_refusal_rate": over_refusal,
        },
    }


def test_acceptance_requires_safety_improvement_without_utility_regression():
    before = _report(attack=0.10, leak=0.60, completion=0.90, over_refusal=0.10)
    after = _report(attack=0.05, leak=0.45, completion=0.87, over_refusal=0.13)
    result = evaluate_acceptance(before, after)
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_attack_regression_rejects_candidate_even_when_benign_improves():
    before = _report(attack=0.0, leak=0.60, completion=0.90, over_refusal=0.10)
    after = _report(attack=0.05, leak=0.20, completion=0.90, over_refusal=0.10)
    result = evaluate_acceptance(before, after)
    assert result["passed"] is False
    assert result["checks"]["attack_not_worse"] is False


def test_zero_baseline_leak_requires_no_regression_not_negative_rate():
    report = _report(attack=0.0, leak=0.0, completion=1.0, over_refusal=0.0)
    result = evaluate_acceptance(report, report)
    assert result["passed"] is True
    assert result["thresholds"]["benign_leak_target"] == 0.0


def test_post_completion_regression_rejects_candidate_even_if_overall_leak_improves():
    before = _report(attack=0.10, leak=0.60, completion=0.90, over_refusal=0.10)
    after = _report(attack=0.05, leak=0.40, completion=0.90, over_refusal=0.10)
    before["overall_benign"].update({
        "leak_before_completion_rate": 0.30,
        "leak_at_completion_rate": 0.20,
        "post_completion_leak_rate": 0.10,
    })
    after["overall_benign"].update({
        "leak_before_completion_rate": 0.10,
        "leak_at_completion_rate": 0.10,
        "post_completion_leak_rate": 0.20,
    })
    result = evaluate_acceptance(before, after)
    assert result["passed"] is False
    assert result["checks"]["post_completion_leak_not_worse"] is False
