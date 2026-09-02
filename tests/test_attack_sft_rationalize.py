from redteamrl.attack_sft.rationalize import (
    build_authoring_prompt,
    author_turn,
    WIN_FRAMING,
    ROUTINE_FRAMING,
    BOUNDARY_FRAMING,
)


def test_authoring_prompt_contains_observation_and_action_not_the_future():
    p = build_authoring_prompt("saw config.py listing", {"kind": "tool_call", "tool": "read_file"}, WIN_FRAMING)
    assert "config.py" in p and "read_file" in p


def test_nearmiss_framing_is_retired():
    """Episode-level near-miss framing is gone (Task 5): pipeline.py is turn-level now, and
    boundary turns use BOUNDARY_FRAMING for both allowed-after-prior-failure and rejected turns."""
    import redteamrl.attack_sft.rationalize as R
    assert not hasattr(R, "NEARMISS_FRAMING")


def test_author_turn_returns_reasoning_text():
    out = author_turn("obs", {"kind": "message", "text": "x"}, WIN_FRAMING, lambda s, m: "because it fits the cover")
    assert out == "because it fits the cover"


def test_three_framings_exist():
    import redteamrl.attack_sft.rationalize as R
    assert R.ROUTINE_FRAMING and R.BOUNDARY_FRAMING and R.WIN_FRAMING


def test_routine_framing_asks_for_a_very_short_reason():
    low = ROUTINE_FRAMING.lower()
    assert "quick" in low or "brief" in low or "short" in low or "few words" in low


def test_boundary_framing_asks_to_infer_the_boundary_from_observed_failure():
    low = BOUNDARY_FRAMING.lower()
    assert ("denied" in low or "redact" in low or "blocked" in low or "rejected" in low)
    assert "infer" in low or "boundary" in low or "why" in low


def test_teacher_is_injected_into_the_prompt_but_flagged_not_to_quote():
    p = build_authoring_prompt("obs", {"kind": "tool_call"}, BOUNDARY_FRAMING,
                               teacher="denied: direct credential read")
    assert "direct credential read" in p          # teacher present to guide
    assert "do not quote" in p.lower() or "don't quote" in p.lower() or "your own words" in p.lower()


def test_author_turn_returns_generate_output():
    out = author_turn("obs", {"kind": "message", "text": "x"}, ROUTINE_FRAMING,
                      lambda s, m: "routine read, looks fine")
    assert out == "routine read, looks fine"
