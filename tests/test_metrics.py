from redteamrl.harness.episode import EpisodeResult
from redteamrl.eval.metrics import AttackMetrics, BenignMetrics

def _er(leak=False, complete=False, n_turns=3, complete_at_turn=None):
    return EpisodeResult(
        outcome="leak" if leak else "timeout", n_turns=n_turns,
        complete=complete, leak=leak, complete_at_turn=complete_at_turn,
        attacker_reward=0.0, defender_reward=0.0, steps=[],
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
          _er(complete=True, complete_at_turn=3, leak=True)]   # over-disclosure
    m = BenignMetrics.from_results(rs)
    assert m.completion_rate == 2/3
    assert m.over_refusal_rate == 1/3
    assert m.benign_leak_rate == 1/3
    assert m.avg_steps_to_completion == 2.0    # (1+3)/2

def test_empty_results_do_not_crash():
    assert AttackMetrics.from_results([]).n_runs == 0
    assert BenignMetrics.from_results([]).n_runs == 0