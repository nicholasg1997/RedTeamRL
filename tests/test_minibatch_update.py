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


def test_brake_ceiling_stops_a_runaway_minibatch_the_mean_still_hides():
    from redteamrl.train.train import brake_fires
    # v7.1 overfit iter 0: early minibatches sit near 0, so the running mean lagged while the policy
    # ran away (kl_iter 0.44, ~40x target). Any single minibatch past 10x the target now brakes.
    assert brake_fires([0.0] * 7 + [0.11], target_kl=0.01)         # mean 0.014, but 0.11 > 0.1
    assert not brake_fires([0.002, 0.004, 0.03], target_kl=0.01)  # v7's in-band spike still passes


# --- v7.2: LR warmup over the run's first optimizer steps ---------------------------------------

class _RecordingSGD(torch.optim.SGD):
    """SGD that records the learning rate in force at every step."""
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.seen = []

    def step(self, closure=None):
        self.seen.append(self.param_groups[0]["lr"])
        return super().step(closure)


def test_warmup_ramps_the_lr_over_the_runs_first_steps_and_restores_it():
    # Adam's first steps move every parameter by ~lr whatever the gradient size (v7.1 iter 0: one
    # step, kl 0.054; iter 1: four steps, kl 0.37). A linear ramp over the run's first optimizer
    # steps keeps those steps small while the moment estimates build up.
    model = _tiny_model()
    opt = _RecordingSGD(model.parameters(), lr=0.1)
    m = update_step(Learner(model, None, opt), _examples(), beta=0.0, inner_epochs=2, minibatches=2,
                    shuffle_seed="x", warmup_steps=4, global_step=0)
    assert m["steps"] == 4
    assert opt.seen == pytest.approx([0.025, 0.05, 0.075, 0.1])
    assert opt.param_groups[0]["lr"] == 0.1                  # the base lr is restored afterwards


def test_warmup_resumes_mid_ramp_from_the_global_step():
    model = _tiny_model()
    opt = _RecordingSGD(model.parameters(), lr=0.1)
    update_step(Learner(model, None, opt), _examples(), beta=0.0, inner_epochs=2, minibatches=2,
                shuffle_seed="x", warmup_steps=4, global_step=2)
    assert opt.seen == pytest.approx([0.075, 0.1, 0.1, 0.1])


def test_no_warmup_by_default():
    model = _tiny_model()
    opt = _RecordingSGD(model.parameters(), lr=0.1)
    update_step(Learner(model, None, opt), _examples(), beta=0.0, inner_epochs=2, minibatches=2,
                shuffle_seed="x")
    assert opt.seen == [0.1] * 4
