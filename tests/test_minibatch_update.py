"""v7: opt-in minibatch GRPO steps. Defaults must reproduce today's update exactly (the defender
trainer shares update_step); minibatching takes one optimizer step per minibatch from a frozen
behaviour pass, and the target-KL brake ends an update that moves the policy too far."""
import pytest
import torch

pytest.importorskip("transformers")

from test_batched_update import _examples, _tiny_model  # noqa: E402

from redteamrl.train.capture import Example  # noqa: E402
from redteamrl.train.learner import Learner  # noqa: E402
from redteamrl.train.train import episode_minibatches, update_step  # noqa: E402


def test_episode_minibatches_partition_keeps_episodes_whole():
    live = [Example([1], [2], task_id="t", episode_id=e, advantage=1.0) for e in (0, 0, 1, 2, 2, 3)]
    mbs = episode_minibatches(live, 2, "s")
    assert sorted(i for mb in mbs for i in mb) == list(range(6))
    for mb in mbs:
        eps = {live[i].episode_id for i in mb}
        assert all(live[j].episode_id not in eps for other in mbs if other is not mb for j in other)
    assert episode_minibatches(live, 2, "s") == mbs
    assert len(episode_minibatches(live, 8, "s")) == 4        # never empty minibatches
    assert len(episode_minibatches(live, 2, None)) == 2


def test_defaults_reproduce_todays_update_exactly():
    a, b = _tiny_model(), _tiny_model()
    la = Learner(a, None, torch.optim.SGD(a.parameters(), lr=0.1))
    lb = Learner(b, None, torch.optim.SGD(b.parameters(), lr=0.1))
    ma = update_step(la, _examples(), beta=0.04, inner_epochs=3)
    mb = update_step(lb, _examples(), beta=0.04, inner_epochs=3, minibatches=1, target_kl=None)
    assert ma["epoch_ratios"] == mb["epoch_ratios"] and mb["steps"] == 3
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pb, pa)


def test_minibatches_take_one_step_each_and_report_kl():
    model = _tiny_model()
    learner = Learner(model, None, torch.optim.SGD(model.parameters(), lr=0.1))
    m = update_step(learner, _examples(), beta=0.0, inner_epochs=2, minibatches=2,
                    shuffle_seed="x")
    assert m["steps"] == 4 and not m["early_stopped"]
    assert m["minibatch_kl_old"][0] == pytest.approx(0.0, abs=1e-7)   # behaviour weights: no drift
    assert m["kl_iter"] > 0 and 0.0 <= m["clip_frac"] <= 1.0


def test_target_kl_brake_stops_the_update():
    model = _tiny_model()
    learner = Learner(model, None, torch.optim.SGD(model.parameters(), lr=50.0))
    m = update_step(learner, _examples(), beta=0.0, inner_epochs=2, minibatches=2,
                    target_kl=1e-6, shuffle_seed="x")
    assert m["early_stopped"] and m["steps"] == 1


def test_no_live_examples_returns_the_new_keys():
    exs = [Example([1], [2], task_id="t", episode_id=0, advantage=0.0)]
    m = update_step(Learner(_tiny_model(), None, None), exs, minibatches=8, target_kl=0.01)
    assert (m["steps"], m["early_stopped"], m["kl_iter"], m["clip_frac"]) == (0, False, 0.0, 0.0)


# --- v7.1: the brake judges the epoch's running mean, not one noisy minibatch ---------------------

def test_brake_ignores_a_single_spike_when_the_running_mean_is_in_band():
    from redteamrl.train.train import brake_fires
    # v7 iter 0: mean 0.0065 but one ~10-episode minibatch spiked past 0.02 and stopped the update.
    assert not brake_fires([0.002, 0.004, 0.03], target_kl=0.01)       # mean 0.012 < 0.02
    assert brake_fires([0.02, 0.025, 0.03], target_kl=0.01)             # mean 0.025 > 0.02
    assert not brake_fires([0.5], target_kl=None)
    assert not brake_fires([], target_kl=0.01)
