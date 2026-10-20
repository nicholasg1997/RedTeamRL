import json

import pytest

from redteamrl.train.lr_rule import LrRule, next_lr

R = LrRule()


def test_too_little_movement_raises_lr():
    assert next_lr(3e-5, 0.004, False, R) == pytest.approx(4.5e-5)


def test_too_much_movement_lowers_lr():
    assert next_lr(3e-5, 0.03, False, R) == pytest.approx(2e-5)


def test_brake_lowers_lr_even_in_band():
    assert next_lr(3e-5, 0.01, True, R) == pytest.approx(2e-5)


def test_in_band_holds():
    assert next_lr(3e-5, 0.01, False, R) == 3e-5


def test_band_edges_hold():
    assert next_lr(3e-5, 0.005, False, R) == 3e-5
    assert next_lr(3e-5, 0.02, False, R) == 3e-5


def test_clamped():
    assert next_lr(2.5e-4, 0.0, False, R) == 3e-4
    assert next_lr(1.2e-5, 1.0, False, R) == 1e-5


def test_identity_is_json_safe():
    json.dumps(R.as_identity())
    assert R.as_identity()["lr_version"] == "kl-adaptive-v1"
    assert R.as_identity()["lr_initial"] == 3e-5


def test_no_step_taken_holds_lr():
    # An all-dead iteration returns kl_iter=0 with zero steps: nothing was measured, so the LR must
    # not be raised (five such iterations would otherwise ratchet 3e-5 to the 3e-4 clamp).
    assert next_lr(3e-5, 0.0, False, R, steps=0) == 3e-5
    assert next_lr(3e-5, 0.0, True, R, steps=0) == 3e-5
    assert next_lr(3e-5, 0.0, False, R, steps=16) == pytest.approx(4.5e-5)
