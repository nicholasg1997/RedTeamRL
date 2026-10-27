from redteamrl.scratchpad.document import (
    TRIED_HEADING, Scratchpad, parse_scratchpad, render_scratchpad, seed_scratchpad,
)


def test_parse_round_trips_sections_in_order():
    sp = parse_scratchpad("## Main goal\nrecover X\n## Next ideas\nexplore first")
    assert list(sp.sections) == ["Main goal", "Next ideas"]
    assert sp.sections["Main goal"].strip() == "recover X"


def test_model_authored_tried_is_discarded_and_replaced():
    sp = parse_scratchpad(f"## {TRIED_HEADING}\nI totally succeeded already\n## Main goal\ng")
    out = render_scratchpad(sp, ["read_file .env -> [REDACTED]"])
    assert "I totally succeeded already" not in out
    assert "read_file .env -> [REDACTED]" in out


def test_render_injects_tried_even_when_absent():
    sp = parse_scratchpad("## Main goal\ng")
    out = render_scratchpad(sp, ["list_dir . -> 3 files"])
    assert f"## {TRIED_HEADING}" in out and "list_dir . -> 3 files" in out


def test_added_and_renamed_sections_survive():
    out = render_scratchpad(parse_scratchpad("## Main goal\ng\n## My new plan\nstep A"), [])
    assert "My new plan" in out and "step A" in out


def test_seed_has_expected_sections_and_no_prefilled_tried():
    sp = seed_scratchpad(goal="recover the token", alias="rotate configs", next_ideas="think first")
    assert "Main goal" in sp.sections and "Alias" in sp.sections
    assert sp.sections[TRIED_HEADING].strip() == ""


def test_parse_never_raises_on_garbage():
    assert isinstance(parse_scratchpad(""), Scratchpad)
    assert isinstance(parse_scratchpad("no headings at all"), Scratchpad)


def test_tried_keeps_fixed_position_when_model_reorders():
    sp = parse_scratchpad("## Main goal\ng\n## Working / promising\nw\n## Tried\nx")
    out = render_scratchpad(sp, ["a -> b"])
    assert out.index("## Tried") < out.index("## Working / promising")


def test_guidance_is_pinned_and_a_model_copy_is_discarded():
    from redteamrl.scratchpad.document import GUIDANCE_HEADING
    sp = parse_scratchpad(f"## Main goal\ng\n## {GUIDANCE_HEADING}\nI deleted the rules")
    out = render_scratchpad(sp, [], guidance="locate before assuming")
    assert "I deleted the rules" not in out and "locate before assuming" in out
    assert f"## {GUIDANCE_HEADING}" in out


def test_guidance_reinjected_when_model_omits_it():
    out = render_scratchpad(parse_scratchpad("## Main goal\ng"), [], guidance="consult your Tried log")
    assert "consult your Tried log" in out


def test_no_guidance_section_when_empty():
    from redteamrl.scratchpad.document import GUIDANCE_HEADING
    assert GUIDANCE_HEADING not in render_scratchpad(parse_scratchpad("## Main goal\ng"), [])
