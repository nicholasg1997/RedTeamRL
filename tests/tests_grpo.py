import math
import torch

from redteamrl.train.grpo import group_advantages
from redteamrl.train.grpo import k3_kl, example_loss


def test_zscore_within_task():
    # one task, rewards [1, -2, 1, 1] -> mean -0.25 ... just check signs + zero-mean-ish
    adv = group_advantages([1.0, -2.0, 1.0, 1.0], ["t", "t", "t", "t"])
    assert adv[1] < 0 < adv[0]                      # the -2 is below the group mean
    assert abs(sum(adv)) < 1e-6                      # z-scores are zero-mean


def test_dead_group_is_zero():
    # all identical -> std 0 -> advantage 0 (no gradient), never NaN
    adv = group_advantages([1.0, 1.0, 1.0], ["t", "t", "t"])
    assert adv == [0.0, 0.0, 0.0]


def test_groups_are_independent():
    # task A varies, task B is dead; interleaved order must be respected
    adv = group_advantages([1.0, -1.0, 5.0, 5.0], ["A", "A", "B", "B"])
    assert adv[0] > 0 > adv[1]      # A varies
    assert adv[2] == 0 and adv[3] == 0   # B dead


def test_k3_is_nonneg_and_zero_when_equal():
    lp = torch.tensor([-0.5, -1.0, -2.0])
    assert torch.allclose(k3_kl(lp, lp), torch.zeros(3), atol=1e-6)   # identical -> 0
    kl = k3_kl(new_lp=torch.tensor([-2.0]), ref_lp=torch.tensor([-0.5]))
    assert (kl >= 0).all()                                            # always >= 0


def test_ratio_is_one_when_old_equals_new():
    # mu=1 case: old==new -> ratio 1 -> surrogate == advantage * logprob-grad; loss finite
    lp = torch.tensor([-0.5, -1.0], requires_grad=True)
    loss = example_loss(new_lp=lp, old_lp=lp.detach(), ref_lp=lp.detach(),
                        advantage=1.0, mask=torch.tensor([1.0, 1.0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert lp.grad is not None                                        # gradient flows


def test_mask_zeros_prompt_tokens():
    lp = torch.tensor([-0.5, -1.0, -2.0], requires_grad=True)
    full = example_loss(lp, lp.detach(), lp.detach(), 1.0, torch.tensor([1.0, 1.0, 1.0]))
    half = example_loss(lp, lp.detach(), lp.detach(), 1.0, torch.tensor([1.0, 0.0, 0.0]))
    assert not torch.allclose(full, half)     # masking changes the loss