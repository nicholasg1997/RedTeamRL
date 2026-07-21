from types import SimpleNamespace
from redteamrl.train.capture import Example
from redteamrl.train.train import assign_advantages, _tally_invalid
from redteamrl.harness.protocol import _INVALID_ATTACKER_OUTPUT, _DEFENDER_FAIL_MARKER


def test_tally_invalid_localizes_defender_and_agent_failures():
    result = SimpleNamespace(n_turns=3, steps=[
        # valid attacker call; defender allow (valid) + result pass (valid)
        {"action": {"kind": "tool_call", "tool": "run_command"},
         "call_decision": {"reasoning": "ok", "verdict": "allow"},
         "result_decision": {"reasoning": "ok", "verdict": "pass"}},
        # INVALID attacker action (fallback message) + a FAIL-CLOSED defender decision
        {"action": {"kind": "message", "text": _INVALID_ATTACKER_OUTPUT},
         "call_decision": {"reasoning": _DEFENDER_FAIL_MARKER + " garbage", "verdict": "reject"},
         "result_decision": None},
    ])
    diag = {"def_dec": 0, "def_fail": 0, "atk_act": 0, "atk_inv": 0, "ben_act": 0, "ben_inv": 0, "turns": []}
    _tally_invalid(diag, "attack", result)
    assert diag["atk_act"] == 2 and diag["atk_inv"] == 1     # one invalid attacker action of two
    assert diag["def_dec"] == 3 and diag["def_fail"] == 1    # allow+pass+reject; one fail-closed
    assert diag["ben_act"] == 0 and diag["turns"] == [3]


def test_assign_advantages_broadcasts_episode_reward():
    # task T, 2 episodes: ep0 reward +1 (2 decisions), ep1 reward -2 (1 decision)
    exs = [
        Example([1], [2], task_id="T", episode_id=0, reward=1.0),
        Example([1], [3], task_id="T", episode_id=0, reward=1.0),
        Example([1], [4], task_id="T", episode_id=1, reward=-2.0),
    ]
    assign_advantages(exs)
    # both decisions of ep0 share one advantage; ep1's is the opposite sign
    assert exs[0].advantage == exs[1].advantage
    assert exs[0].advantage > 0 > exs[2].advantage


def test_assign_advantages_dead_task_is_zero():
    exs = [
        Example([1], [2], task_id="T", episode_id=0, reward=1.0),
        Example([1], [3], task_id="T", episode_id=1, reward=1.0),
    ]
    assign_advantages(exs)
    assert all(e.advantage == 0.0 for e in exs)