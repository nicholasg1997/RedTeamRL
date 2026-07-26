from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import redteamrl.train.train as train_mod
from redteamrl.train.train import rollout


class _Cap:
    """Mirrors CapturingGenerate's episode-scoped capture interface."""

    def __init__(self):
        self.buffer = []
        self._scoped = None

    @contextmanager
    def episode_capture(self):
        self._scoped = []
        try:
            yield self._scoped
        finally:
            self._scoped = None

    def emit(self):
        sink = self.buffer if self._scoped is None else self._scoped
        sink.append(SimpleNamespace(task_id=None, episode_id=None, reward=None, advantage=0.0))


def _spec(task_id, episode_type):
    return SimpleNamespace(id=task_id, episode_type=episode_type)


def _patched(monkeypatch, cap, defender_reward, attacker_reward):
    def fake_run_episode(spec, agent, defender, sandbox, max_turns,
                         post_completion_turns=None, redaction_enforcement="unshielded"):
        cap.emit()
        return SimpleNamespace(defender_reward=defender_reward, attacker_reward=attacker_reward,
                               n_turns=3, steps=[])
    monkeypatch.setattr(train_mod, "run_episode", fake_run_episode)


def _run(monkeypatch, trained_side):
    cap = _Cap()
    _patched(monkeypatch, cap, defender_reward=-2.0, attacker_reward=0.9)
    return rollout(
        [_spec("t-attack", "attack")],
        lambda spec: None, lambda spec: None, lambda: None,
        lambda: SimpleNamespace(close=lambda: None),
        cap, n_rollouts=2, trained_side=trained_side,
    )


def test_defender_side_tags_examples_with_defender_reward(monkeypatch):
    assert [ex.reward for ex in _run(monkeypatch, "defender")] == [-2.0, -2.0]


def test_attacker_side_tags_examples_with_attacker_reward(monkeypatch):
    # The attacker phase reuses the same loop; only which reward labels the captured examples
    # changes. Without this the attacker would be trained on the defender's objective.
    assert [ex.reward for ex in _run(monkeypatch, "attacker")] == [0.9, 0.9]


def test_unknown_side_is_rejected_rather_than_silently_defaulting(monkeypatch):
    with pytest.raises(ValueError, match="trained_side"):
        _run(monkeypatch, "both")


def test_verdicts_are_captured_for_diversity(monkeypatch):
    # group_diversity was receiving empty verdict lists, so mixed_verdict_group_rate and
    # verdict_entropy were structurally zero — the constant-policy signal measured nothing.
    cap = _Cap()

    def fake_run_episode(spec, agent, defender, sandbox, max_turns,
                         post_completion_turns=None, redaction_enforcement="unshielded"):
        cap.emit()
        return SimpleNamespace(
            defender_reward=-2.0, attacker_reward=0.9, n_turns=2,
            steps=[{"action": {"kind": "tool_call"},
                    "call_decision": {"verdict": "allow"},
                    "result_decision": {"verdict": "pass"}}],
        )

    monkeypatch.setattr(train_mod, "run_episode", fake_run_episode)
    examples = rollout(
        [_spec("t-attack", "attack")],
        lambda spec: None, lambda spec: None, lambda: None,
        lambda: SimpleNamespace(close=lambda: None),
        cap, n_rollouts=1, trained_side="defender",
    )
    assert examples[0].verdicts == [
        {"decision_type": "call", "verdict": "allow"},
        {"decision_type": "result", "verdict": "pass"},
    ]
