from redteamrl.harness.episode import EpisodeResult
from redteamrl.eval.metrics import AttackMetrics, BenignMetrics

def _er(leak=False, complete=False, n_turns=3, complete_at_turn=None,
        leak_phase=None, post_completion_turns=0, **kwargs):
    return EpisodeResult(
        outcome="leak" if leak else "timeout", n_turns=n_turns,
        complete=complete, leak=leak, complete_at_turn=complete_at_turn,
        leak_phase=leak_phase, post_completion_turns=post_completion_turns,
        attacker_reward=0.0, defender_reward=0.0, steps=[],
        **kwargs,
    )

def test_attack_metrics():
    rs = [_er(leak=True, n_turns=2), _er(leak=False), _er(leak=True, n_turns=4), _er(leak=True, n_turns=6)]
    m = AttackMetrics.from_results(rs)
    assert m.attack_success_rate == 0.75
    assert m.avg_steps_to_leak == 4.0          # (2+4+6)/3
    assert m.n_runs == 4

def test_attack_metrics_no_leaks():
    m = AttackMetrics.from_results([_er(leak=False), _er(leak=False)])
    assert m.attack_success_rate == 0.0
    assert m.avg_steps_to_leak is None

def test_benign_metrics():
    rs = [_er(complete=True, complete_at_turn=1),
          _er(complete=False),
          _er(complete=True, complete_at_turn=3, leak=True,
              leak_phase="at_completion")]   # over-disclosure
    m = BenignMetrics.from_results(rs)
    assert m.completion_rate == 2/3
    assert m.over_refusal_rate == 1/3
    assert m.benign_leak_rate == 1/3
    assert m.leak_before_completion_rate == 0.0
    assert m.leak_at_completion_rate == 1/3
    assert m.post_completion_leak_rate == 0.0
    assert m.avg_steps_to_completion == 2.0    # (1+3)/2


def test_benign_metrics_split_post_completion_leaks_and_exposure():
    rs = [
        _er(complete=False, leak=True, leak_phase="pre_completion"),
        _er(complete=True, complete_at_turn=1, leak=True,
            leak_phase="at_completion"),
        _er(complete=True, complete_at_turn=1, leak=True,
            leak_phase="post_completion", post_completion_turns=2),
        _er(complete=True, complete_at_turn=1, post_completion_turns=4),
    ]
    m = BenignMetrics.from_results(rs)
    assert m.benign_leak_rate == 3/4
    assert m.leak_before_completion_rate == 1/4
    assert m.leak_at_completion_rate == 1/4
    assert m.post_completion_leak_rate == 1/4
    assert m.conditional_post_completion_leak_rate == 1/2
    assert m.post_completion_leaks_per_100_turns == 100/6
    assert m.n_post_completion_exposed == 2
    assert m.avg_post_completion_turns == 3.0


def test_empty_results_do_not_crash():
    assert AttackMetrics.from_results([]).n_runs == 0
    benign = BenignMetrics.from_results([])
    assert benign.n_runs == 0
    assert benign.conditional_post_completion_leak_rate is None
    assert benign.post_completion_leaks_per_100_turns is None


def test_policy_leaks_and_redaction_failures_are_separate_from_delivery():
    results = [
        _er(
            complete=False,
            policy_leak=True,
            policy_leak_at_turn=0,
            failed_redactions=1,
            redaction_attempts=1,
            redaction_opportunities=1,
            result_decisions=1,
        ),
        _er(
            complete=True,
            complete_at_turn=1,
            effective_redactions=1,
            redaction_attempts=1,
            redaction_opportunities=1,
            result_decisions=1,
        ),
        _er(
            complete=True,
            complete_at_turn=1,
            unnecessary_redactions=1,
            redaction_attempts=1,
            result_decisions=1,
        ),
        _er(
            leak=True,
            complete=True,
            complete_at_turn=1,
            leak_phase="at_completion",
            sensitive_result_passes=1,
            redaction_opportunities=1,
            result_decisions=1,
        ),
    ]

    metrics = BenignMetrics.from_results(results)

    assert metrics.benign_leak_rate == 1 / 4
    assert metrics.policy_leak_rate == 2 / 4
    assert metrics.shielded_policy_leak_rate == 1 / 4
    assert metrics.policy_safe_completion_rate == 2 / 4
    assert metrics.delivered_safe_completion_rate == 2 / 4
    assert metrics.redaction.redaction_opportunities == 3
    assert metrics.redaction.effective_redactions == 1
    assert metrics.redaction.failed_redactions == 1
    assert metrics.redaction.sensitive_result_passes == 1
    assert metrics.redaction.failed_redaction_rate == 1 / 2
    assert metrics.redaction.sensitive_result_pass_rate == 1 / 3
    assert metrics.redaction.unnecessary_redaction_rate == 1 / 3
