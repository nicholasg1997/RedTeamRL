from types import SimpleNamespace
import pytest
import torch

from redteamrl.train.capture import Example
from redteamrl.train.train import assign_advantages, _tally_invalid, update_step
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


class _ScalarLearner:
    """Minimal differentiable learner for testing update aggregation without model weights."""

    def __init__(self):
        self.value = torch.nn.Parameter(torch.tensor(0.0))
        self.model = torch.nn.Module()
        self.model.register_parameter("value", self.value)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def logprobs(self, prompt_ids, completion_ids, use_adapter, with_grad):
        if use_adapter and with_grad:
            return self.value.expand(len(completion_ids))
        return torch.zeros(len(completion_ids))


def _one_update(n_decisions: int) -> tuple[float, dict]:
    learner = _ScalarLearner()
    examples = [
        Example([1], [2], task_id="T", episode_id=0, reward=1.0, advantage=1.0)
        for _ in range(n_decisions)
    ]
    metrics = update_step(learner, examples, beta=0.0)
    return learner.value.item(), metrics


def test_update_is_invariant_to_duplicate_decisions_within_an_episode():
    one_value, one_metrics = _one_update(1)
    many_value, many_metrics = _one_update(8)

    assert many_value == one_value
    assert one_metrics["n_live_episodes"] == many_metrics["n_live_episodes"] == 1
    assert one_metrics["n_live_examples"] == 1
    assert many_metrics["n_live_examples"] == 8


def test_update_skips_optimizer_cleanly_when_every_group_is_dead():
    learner = _ScalarLearner()
    metrics = update_step(
        learner,
        [Example([1], [2], task_id="T", episode_id=0, advantage=0.0)],
    )

    assert learner.value.item() == 0.0
    assert metrics["loss"] == 0.0
    assert metrics["grad_norm"] == 0.0
    assert metrics["n_live_episodes"] == 0


def test_update_weights_episodes_equally_without_retaining_every_graph():
    # The mean-of-means aggregation held one autograd graph per live example until a single
    # backward() at the end. At iteration 0 that was 350 graphs through a 4B model and OOM'd an
    # 80GB A100. Weighting each loss by 1/(n_episodes * decisions_in_episode) and backpropagating
    # immediately gives an IDENTICAL gradient with O(1) graphs alive.
    learner = _ScalarLearner()
    examples = [
        *[Example([1], [2], task_id="T", episode_id=0, advantage=1.0) for _ in range(3)],
        Example([1], [2], task_id="T", episode_id=1, advantage=3.0),
    ]

    backward_calls = []
    real_backward = torch.Tensor.backward

    def counting_backward(self, *args, **kwargs):
        backward_calls.append(1)
        return real_backward(self, *args, **kwargs)

    torch.Tensor.backward = counting_backward
    try:
        metrics = update_step(learner, examples, beta=0.0)
    finally:
        torch.Tensor.backward = real_backward

    # One backward per live example -- each graph is freed before the next is built.
    assert len(backward_calls) == 4
    # Gradient is unchanged: mean over episodes of (mean over decisions) = (-1 + -3)/2 = -2,
    # so SGD(lr=0.1) moves the parameter to +0.2. A three-decision episode must not outweigh a
    # one-decision episode.
    assert learner.value.item() == pytest.approx(0.2, abs=1e-6)
    assert metrics["n_live_episodes"] == 2
    assert metrics["n_live_examples"] == 4
    assert metrics["loss"] == pytest.approx(-2.0, abs=1e-6)
