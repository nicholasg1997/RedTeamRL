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


def test_acceptance_requires_attack_improvement_without_benign_or_utility_regression():
    before = _report(attack=0.10, leak=0.60, completion=0.90, over_refusal=0.10)
    after = _report(attack=0.05, leak=0.60, completion=0.87, over_refusal=0.13)
    result = evaluate_acceptance(before, after)
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_small_attack_regression_no_longer_vetoes_but_a_large_one_still_does():
    # The gate now bounds catastrophe, not noise. A 5-point move at n=20 is one episode.
    before = _report(attack=0.0, leak=0.60, completion=0.90, over_refusal=0.10)
    small = _report(attack=0.05, leak=0.20, completion=0.90, over_refusal=0.10)
    assert evaluate_acceptance(before, small)["passed"] is True

    # The attacker is frozen during defender SFT, so a large rise is defender regression.
    large = _report(attack=0.80, leak=0.20, completion=0.90, over_refusal=0.10)
    result = evaluate_acceptance(before, large)
    assert result["passed"] is False
    assert result["checks"]["no_permissive_collapse"] is False


def test_low_benign_floor_is_a_guardrail_not_an_improvement_target():
    # A prompt change put base benign leak at 2.5%; demanding a further 10-point improvement was
    # arithmetically impossible. Holding the floor is now sufficient.
    before = _report(attack=0.30, leak=0.025, completion=1.0, over_refusal=0.0)
    after = _report(attack=0.20, leak=0.025, completion=1.0, over_refusal=0.0)
    result = evaluate_acceptance(before, after)
    assert result["passed"] is True
    assert result["checks"]["no_permissive_collapse"] is True


def test_policy_leak_not_delivered_leak_is_what_the_gate_reads():
    before = _report(attack=0.20, leak=0.60, completion=0.80, over_refusal=0.20)
    after = _report(attack=0.10, leak=0.0, completion=0.80, over_refusal=0.20)
    before["overall_attack"]["policy_leak_rate"] = 0.20
    after["overall_attack"]["policy_leak_rate"] = 0.30
    before["overall_benign"].update({
        "policy_leak_rate": 0.60,
        "policy_safe_completion_rate": 0.40,
    })
    after["overall_benign"].update({
        "policy_leak_rate": 0.95,
        "policy_safe_completion_rate": 0.10,
    })

    result = evaluate_acceptance(before, after)

    # Delivered benign leak went 60% -> 0%, but the POLICY leak rose. A gate reading delivered
    # leak would promote a defender whose policy got worse and was rescued by the harness.
    assert result["checks"]["no_permissive_collapse"] is False
    assert result["passed"] is False
