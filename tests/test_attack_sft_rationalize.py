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


def test_near_miss_framing_is_retired():
    """Episode-level near-miss framing is gone (Task 5): pipeline.py is turn-level now, and
    boundary turns use BOUNDARY_FRAMING for both allowed-after-prior-failure and rejected turns."""
    import redteamrl.attack_sft.rationalize as R
    assert not hasattr(R, "NEARMISS_FRAMING")


def test_author_turn_keeps_the_real_action_and_strips_stray_json_from_reasoning():
    # The teacher authors REASONING; the turn's REAL action is kept (no synthesized replacement).
    # If the model emits a trailing JSON out of habit, it is stripped from the reasoning prose.
    res, act = author_turn("obs", {"kind": "message", "text": "x"}, WIN_FRAMING,
                           lambda s, m: 'because it fits\n{"kind": "message", "text": "y"}')
    assert res == "because it fits"
    assert act == {"kind": "message", "text": "x"}     # ORIGINAL action, not the teacher's JSON


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


def test_author_turn_fallback_when_no_json_emitted():
    res, act = author_turn("obs", {"kind": "tool_call", "tool": "read_file", "args": {"path": "a"}}, ROUTINE_FRAMING,
                           lambda s, m: "routine read, looks fine")
    assert res == "routine read, looks fine"
    assert act == {"kind": "tool_call", "tool": "read_file", "args": {"path": "a"}}


def test_boundary_framing_forbids_empty_route_around_and_demands_a_different_action():
    low = BOUNDARY_FRAMING.lower()
    assert "repeating" in low                      # teaches: repeating changes nothing
    assert "different" in low                      # demands a concrete different technique
    assert "next action" in low                    # ends by naming the next action
    assert "infer" in low and "observed" in low    # teaches: infer the mechanism from observation
    assert "route around it" not in low            # the old vague phrasing is gone


def test_routine_framing_stays_short():
    assert "few words" in ROUTINE_FRAMING.lower()


def test_win_framing_ends_by_naming_the_next_action():
    assert WIN_FRAMING.strip().lower().endswith("next action.")
