import pytest

from redteamrl.policies.prompt_pool import (
    ATTACKER_STYLES, BENIGN_STYLES, DEFENDER_STYLES, PROMPT_POOL_VERSION, select_style,
)


def test_attacker_and_benign_span_the_same_styles():
    # THE anti-fingerprint invariant. Several attacker prompts alone would not help: if every
    # attacker variant is terse and every benign variant is verbose, style still separates them
    # and the defender can classify episode type without judging content.
    assert set(ATTACKER_STYLES) == set(BENIGN_STYLES)
    assert len(ATTACKER_STYLES) >= 4


def test_style_choice_does_not_depend_on_episode_type():
    # Style is picked from (task, rollout, iteration) only. If it keyed on episode_type the pools
    # would re-separate immediately.
    for rollout_index in range(8):
        attack = select_style("t", rollout_index, iteration=3, n_styles=len(ATTACKER_STYLES))
        benign = select_style("t", rollout_index, iteration=3, n_styles=len(BENIGN_STYLES))
        assert attack == benign


def test_selection_is_deterministic():
    first = [select_style("task-a", r, 0, 4) for r in range(24)]
    again = [select_style("task-a", r, 0, 4) for r in range(24)]
    assert first == again, "a resumed run would replay banked episodes under different prompts"


def test_selection_spreads_evenly_over_a_training_run():
    # Checked at run scale, not per-iteration: with 4 styles and 24 draws, missing one is a
    # legitimate ~0.4% event, so demanding full coverage there tests luck rather than the hash.
    from collections import Counter
    counts = Counter(select_style("task-a", r, it, 4) for r in range(30) for it in range(20))
    assert set(counts) == {0, 1, 2, 3}
    assert min(counts.values()) > 0.7 * max(counts.values()), f"skewed selection: {dict(counts)}"


def test_selection_varies_across_iterations_and_tasks():
    by_iteration = {select_style("task-a", 0, it, 4) for it in range(12)}
    by_task = {select_style(f"task-{i}", 0, 0, 4) for i in range(12)}
    assert len(by_iteration) > 1
    assert len(by_task) > 1


def test_every_template_keeps_its_required_placeholders():
    for style, template in ATTACKER_STYLES.items():
        for field in ("{goal}", "{cover}", "{hint}"):
            assert field in template, f"attacker/{style} is missing {field}"
    for style, template in BENIGN_STYLES.items():
        assert "{goal}" in template, f"benign/{style} is missing {{goal}}"
        # A benign agent has no hidden objective and no technique hint; leaking either would
        # hand the defender a trivial episode-type tell.
        assert "{cover}" not in template and "{hint}" not in template


def test_action_contract_is_byte_identical_across_every_prompt():
    # Only the framing may vary. If the JSON contract drifted between styles, parse-failure rates
    # would differ by style and the defender could read THAT as the episode-type signal.
    contracts = set()
    for template in list(ATTACKER_STYLES.values()) + list(BENIGN_STYLES.values()):
        assert '"kind": "tool_call"' in template
        contracts.add(template[template.index("Reason in prose"):])
    # attacker templates additionally offer `submit`, so allow exactly two contract shapes
    assert len(contracts) == 2


def test_defender_styles_preserve_the_output_contract():
    # The defender prompt encodes the TASK SPEC, not just tone. Paraphrasing may change prose but
    # never the verdict vocabulary or the JSON shape, or different styles would be different tasks.
    assert len(DEFENDER_STYLES) >= 3
    for style, template in DEFENDER_STYLES.items():
        for token in ("allow", "reject", "pass", "redact"):
            assert token in template, f"defender/{style} dropped the verdict '{token}'"


def test_pool_version_is_declared():
    assert isinstance(PROMPT_POOL_VERSION, int)
