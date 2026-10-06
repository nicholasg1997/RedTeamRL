"""Batched scoring must change the update's SPEED, never its math.

`Learner.logprobs_batch` left-pads a micro-batch into one forward pass. These tests pin it to the
one-at-a-time path on a real (tiny, randomly initialised, CPU, float32) transformer: identical
log-probs, identical gradients, and an identical model after a full multi-epoch `update_step`.
"""
import contextlib
import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from redteamrl.train.capture import Example
from redteamrl.train.grpo import group_advantages
from redteamrl.train.learner import Learner
from redteamrl.train.train import micro_batches, update_step


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, max_position_embeddings=128)
    model = transformers.LlamaForCausalLM(config).float()
    model.disable_adapter = lambda: contextlib.nullcontext()   # no peft offline; ref = same model
    return model


# Different prompt AND completion lengths, so padding and tail alignment are both exercised.
PAIRS = [([5, 9, 13, 2], [7, 7, 1]), ([3, 1], [8]), ([11, 4, 6, 6, 2, 9, 1], [2, 5, 3, 3, 1]),
         ([10, 20, 30], [40, 50])]


def test_batched_logprobs_equal_one_at_a_time():
    learner = Learner(_tiny_model(), tokenizer=None, optimizer=None)
    single = [learner.logprobs(p, c, use_adapter=True, with_grad=False) for p, c in PAIRS]
    batched = learner.logprobs_batch(PAIRS, use_adapter=True, with_grad=False)
    for s, b in zip(single, batched):
        torch.testing.assert_close(b, s, atol=1e-5, rtol=1e-5)


def test_batched_gradients_equal_one_at_a_time():
    a, b = _tiny_model(), _tiny_model()
    la, lb = Learner(a, None, None), Learner(b, None, None)
    sum(lp.sum() for lp in (la.logprobs(p, c, True, True) for p, c in PAIRS)).backward()
    sum(lp.sum() for lp in lb.logprobs_batch(PAIRS, True, True)).backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pb.grad, pa.grad, atol=1e-5, rtol=1e-4)


class _Unbatched:
    """Exposes ONLY the single-example API, so update_step takes its one-at-a-time path (the
    pre-batching code). A Learner subclass would not do: it inherits logprobs_batch."""

    def __init__(self, model, optimizer):
        self._inner = Learner(model, None, optimizer)
        self.model, self.optimizer = model, optimizer

    def logprobs(self, *args, **kwargs):
        return self._inner.logprobs(*args, **kwargs)


def _examples():
    rewards = [1.0, -0.3, 0.2, -0.5]
    adv = group_advantages(rewards, ["t"] * 4, normalize_std=False)
    return [Example(prompt_ids=p, completion_ids=c, task_id="t", episode_id=i, advantage=a)
            for i, ((p, c), a) in enumerate(zip(PAIRS, adv))]


def test_update_step_is_identical_batched_or_not():
    a, b = _tiny_model(), _tiny_model()
    la = _Unbatched(a, torch.optim.SGD(a.parameters(), lr=0.1))
    lb = Learner(b, None, torch.optim.SGD(b.parameters(), lr=0.1))
    assert not hasattr(la, "logprobs_batch") and hasattr(lb, "logprobs_batch")
    assert len(micro_batches(_examples(), 64, 64)) == 1     # lb really scores them together
    ma = update_step(la, _examples(), beta=0.04, inner_epochs=3)
    mb = update_step(lb, _examples(), beta=0.04, inner_epochs=3, max_batch_tokens=64,
                     max_batch_logit_rows=64)
    assert mb["epoch_ratios"][0] == pytest.approx(1.0)   # frozen and epoch-0 passes agree
    assert ma["epoch_ratios"][-1] != pytest.approx(1.0)  # the update actually moved the policy
    assert mb["epoch_ratios"] == pytest.approx(ma["epoch_ratios"], abs=1e-5)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pb, pa, atol=1e-5, rtol=1e-4)


def test_micro_batches_cover_every_example_once_and_respect_caps():
    exs = [Example(prompt_ids=[0] * n, completion_ids=[0] * k)
           for n, k in [(10, 2), (50, 5), (12, 3), (200, 40), (11, 1)]]
    batches = micro_batches(exs, max_tokens=64, max_logit_rows=12)
    assert sorted(i for b in batches for i in b) == list(range(len(exs)))
    for b in batches:
        width = max(len(exs[i].prompt_ids) + len(exs[i].completion_ids) for i in b)
        keep = max(len(exs[i].completion_ids) + 1 for i in b)
        assert len(b) == 1 or (len(b) * width <= 64 and len(b) * keep <= 12)


def test_mean_centred_advantages_keep_reward_units():
    # A one-repeat difference (-0.30 vs -0.35) stays a 0.05-sized signal instead of becoming
    # a unit-spread advantage -- the reason the attacker phase drops the std division.
    rewards = [-0.35, -0.35] + [-0.30] * 6
    normed = group_advantages(rewards, ["t"] * 8)
    centred = group_advantages(rewards, ["t"] * 8, normalize_std=False)
    assert max(abs(x) for x in normed) > 1.5
    assert max(abs(x) for x in centred) < 0.05
    assert group_advantages([0.2] * 4, ["t"] * 4, normalize_std=False) == [0.0] * 4   # still dead
