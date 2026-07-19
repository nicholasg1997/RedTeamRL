import torch

from redteamrl.train.grpo import group_advantages, k3_kl, example_loss


def test_zscore_within_task():
    # one task, rewards [1, -2, 1, 1] -> the -2 sits below the group mean
    adv = group_advantages([1.0, -2.0, 1.0, 1.0], ["t", "t", "t", "t"])
    assert adv[1] < 0 < adv[0]                       # the -2 is below the group mean
    assert abs(sum(adv)) < 1e-6                       # z-scores are zero-mean


def test_dead_group_is_zero():
    # all identical -> std 0 -> advantage 0 (no gradient), never NaN
    adv = group_advantages([1.0, 1.0, 1.0], ["t", "t", "t"])
    assert adv == [0.0, 0.0, 0.0]


def test_groups_are_independent():
    # task A varies, task B is dead; interleaved order must be respected
    adv = group_advantages([1.0, -1.0, 5.0, 5.0], ["A", "A", "B", "B"])
    assert adv[0] > 0 > adv[1]                        # A varies
    assert adv[2] == 0 and adv[3] == 0                # B dead


def test_k3_is_nonneg_and_zero_when_equal():
    lp = torch.tensor([-0.5, -1.0, -2.0])
    assert torch.allclose(k3_kl(lp, lp), torch.zeros(3), atol=1e-6)   # identical -> 0
    kl = k3_kl(new_lp=torch.tensor([-2.0]), ref_lp=torch.tensor([-0.5]))
    assert (kl >= 0).all()                                            # always >= 0


def test_ratio_is_one_when_old_equals_new():
    # mu=1 case: old==new -> ratio 1 -> loss finite and gradient flows
    lp = torch.tensor([-0.5, -1.0], requires_grad=True)
    loss = example_loss(new_lp=lp, old_lp=lp.detach(), ref_lp=lp.detach(),
                        advantage=1.0, mask=torch.tensor([1.0, 1.0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert lp.grad is not None                                        # gradient flows


def test_mask_excludes_masked_tokens():
    # old != new so per-token loss VARIES across tokens; masking out tokens 1,2 must
    # change the length-normalized mean. (With old==new the per-token loss is constant,
    # so masking is invisible — that made the old test vacuous.)
    new_lp = torch.tensor([-0.5, -1.0, -2.0], requires_grad=True)
    old_lp = torch.tensor([-0.5, -3.0, -3.0])        # token 0 same; 1,2 differ -> ratio != 1
    ref_lp = new_lp.detach()
    full = example_loss(new_lp, old_lp, ref_lp, 1.0, torch.tensor([1.0, 1.0, 1.0]))
    only0 = example_loss(new_lp, old_lp, ref_lp, 1.0, torch.tensor([1.0, 0.0, 0.0]))
    assert not torch.allclose(full, only0)           # masking changes the loss
