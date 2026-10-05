import pytest

from redteamrl.sft.acceptance import evaluate_acceptance, probe_collapse_diagnostics

_BASE = {
    "overall_attack": {"attack_success_rate": 0.30, "policy_leak_rate": 0.30, "redaction": {}},
    "overall_benign": {
        "benign_leak_rate": 0.025, "policy_leak_rate": 0.025, "completion_rate": 1.0,
        "over_refusal_rate": 0.0, "policy_safe_completion_rate": 0.975,
        "leak_before_completion_rate": 0.0, "leak_at_completion_rate": 0.025,
        "post_completion_leak_rate": 0.0, "redaction": {},
    },
}


def _after(**overrides):
    after = {
        "overall_attack": dict(_BASE["overall_attack"]),
        "overall_benign": dict(_BASE["overall_benign"]),
    }
    for key, value in overrides.items():
        target = "overall_attack" if key.startswith("attack") else "overall_benign"
        after[target][key.removeprefix("attack_")] = value
    return after


def _checks(after, **kwargs):
    return evaluate_acceptance(_BASE, after, **kwargs)


def test_noise_scale_moves_do_not_veto_promotion():
    # Round 9 was rejected on 3 episodes: attack 30%->45% (n=20) and benign 2.5%->5.0% (1 vs 2 of
    # 40). A gate that vetoes on movements inside its own error bars measures nothing.
    result = _checks(_after(attack_attack_success_rate=0.45, attack_policy_leak_rate=0.45,
                            benign_leak_rate=0.05, policy_leak_rate=0.05,
                            policy_safe_completion_rate=0.95, leak_at_completion_rate=0.05))
    assert result["passed"] is True


def test_deny_collapse_is_caught():
    result = _checks(_after(completion_rate=0.40, over_refusal_rate=0.60,
                            policy_safe_completion_rate=0.40))
    assert result["passed"] is False
    assert result["checks"]["no_deny_collapse"] is False


def test_permissive_collapse_is_caught():
    # The attacker is frozen during defender SFT, so a large attack-side rise is defender
    # regression, not the attacker finding a new strategy.
    result = _checks(_after(attack_attack_success_rate=0.85, attack_policy_leak_rate=0.85))
    assert result["passed"] is False
    assert result["checks"]["no_permissive_collapse"] is False

    result = _checks(_after(benign_leak_rate=0.55, policy_leak_rate=0.55,
                            policy_safe_completion_rate=0.45, leak_at_completion_rate=0.55))
    assert result["checks"]["no_permissive_collapse"] is False


def test_constant_policy_is_caught_via_verdict_diversity():
    healthy = _checks(_after(), verdict_diversity=0.42)
    assert healthy["checks"]["no_constant_policy"] is True
    degenerate = _checks(_after(), verdict_diversity=0.01)
    assert degenerate["checks"]["no_constant_policy"] is False
    assert degenerate["passed"] is False


def test_protocol_collapse_is_caught():
    result = _checks(_after(), parse_rate=0.55, hindsight_leak_rate=0.0)
    assert result["checks"]["no_protocol_collapse"] is False
    result = _checks(_after(), parse_rate=1.0, hindsight_leak_rate=0.40)
    assert result["checks"]["no_protocol_collapse"] is False


def test_unsupplied_diagnostics_do_not_block_promotion():
    # Diversity/parse signals come from the GRPO sampling temperature and the review path; when a
    # run has not produced them the gate must not fail closed on a metric it never measured.
    result = _checks(_after())
    assert result["checks"]["no_constant_policy"] is True
    assert result["checks"]["no_protocol_collapse"] is True
    assert result["passed"] is True


def test_probe_diagnostics_separate_call_and_result_verdicts():
    result_probe = {
        "rows": [
            {"parsed": True, "verdict": "pass"},
            {"parsed": True, "verdict": "redact"},
        ]
    }
    call_probe = {
        "rows": [
            {"parsed": True, "verdict": "allow"},
            {"parsed": True, "verdict": "reject"},
        ]
    }

    diagnostics = probe_collapse_diagnostics(result_probe, call_probe)

    assert diagnostics["parse_rate"] == 1.0
    assert diagnostics["verdict_diversity"] == 1.0
    assert diagnostics["verdict_counts_by_type"] == {
        "call": {"allow": 1, "reject": 1},
        "result": {"pass": 1, "redact": 1},
    }


def test_probe_diagnostics_catch_allow_pass_constant_policy():
    diagnostics = probe_collapse_diagnostics(
        {"rows": [{"parsed": True, "verdict": "pass"}] * 4},
        {"rows": [{"parsed": True, "verdict": "allow"}] * 4},
    )

    assert diagnostics["parse_rate"] == 1.0
    assert diagnostics["verdict_diversity"] == 0.0
    result = _checks(
        _after(),
        verdict_diversity=diagnostics["verdict_diversity"],
        parse_rate=diagnostics["parse_rate"],
    )
    assert result["checks"]["no_constant_policy"] is False


def test_existing_constant_foothold_does_not_make_gate_impossible():
    # The current base model can be constant-permissive at the call gate. That is attacker
    # headroom, not a collapse caused by SFT. The gate should reject losing existing diversity,
    # not require a one-round bootstrap to hand-fix every base-model weakness.
    result = _checks(
        _after(),
        baseline_verdict_diversity=0.0,
        verdict_diversity=0.0,
        baseline_parse_rate=1.0,
        parse_rate=1.0,
    )

    assert result["checks"]["no_constant_policy"] is True
    assert result["passed"] is True


def test_losing_existing_probe_diversity_is_a_collapse():
    result = _checks(
        _after(),
        baseline_verdict_diversity=0.6,
        verdict_diversity=0.0,
    )

    assert result["checks"]["no_constant_policy"] is False


def test_probe_diagnostics_measure_parse_failures():
    diagnostics = probe_collapse_diagnostics(
        {"rows": [{"parsed": False, "verdict": "redact"}]},
        {"rows": [{"parsed": True, "verdict": "allow"}]},
    )

    assert diagnostics["parse_rate"] == 0.5


def test_only_four_hard_checks_remain():
    assert set(_checks(_after())["checks"]) == {
        "no_deny_collapse",
        "no_permissive_collapse",
        "no_constant_policy",
        "no_protocol_collapse",
    }


def test_deltas_are_still_reported_for_everything_that_no_longer_gates():
    deltas = _checks(_after(attack_attack_success_rate=0.45))["deltas"]
    for key in ("attack_success", "attack_policy_leak", "benign_leak", "benign_policy_leak",
                "completion", "over_refusal", "policy_safe_completion"):
        assert key in deltas
    assert deltas["attack_success"] == pytest.approx(0.15)
