from redteamrl.attack_sft.rationalize import build_authoring_prompt, author_turn, WIN_FRAMING, NEARMISS_FRAMING


def test_authoring_prompt_contains_observation_and_action_not_the_future():
    p = build_authoring_prompt("saw config.py listing", {"kind": "tool_call", "tool": "read_file"}, WIN_FRAMING)
    assert "config.py" in p and "read_file" in p


def test_nearmiss_framing_names_the_block_not_a_fix():
    assert "redact" in NEARMISS_FRAMING.lower() or "block" in NEARMISS_FRAMING.lower()


def test_author_turn_returns_reasoning_text():
    out = author_turn("obs", {"kind": "message", "text": "x"}, WIN_FRAMING, lambda s, m: "because it fits the cover")
    assert out == "because it fits the cover"
