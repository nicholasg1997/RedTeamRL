import pytest

from redteamrl.train.checkpoint import validate_phase_identity


def test_matching_phase_identity_is_accepted():
    identity = {"model": "base", "adapter": "sft-a", "lr": 1e-5}

    validate_phase_identity({"phase_identity": identity}, identity, "/runs/iter2")


def test_mismatched_phase_identity_is_rejected_before_resume():
    with pytest.raises(RuntimeError, match="DIFFERENT phase"):
        validate_phase_identity(
            {"phase_identity": {"model": "base", "adapter": "sft-a"}},
            {"model": "base", "adapter": "sft-b"},
            "/runs/iter2",
        )


def test_missing_legacy_identity_requires_explicit_opt_in():
    with pytest.raises(RuntimeError, match="no phase_identity"):
        validate_phase_identity({"iter": 2}, {"model": "base"}, "/runs/iter2")

    validate_phase_identity(
        {"iter": 2},
        {"model": "base"},
        "/runs/iter2",
        allow_legacy=True,
    )
