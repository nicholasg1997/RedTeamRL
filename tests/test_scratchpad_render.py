from redteamrl.scratchpad.render import action_user_prompt, render_window, scratchpad_user_prompt


def _conv(n):
    conv = [{"role": "context", "content": "task: X"}]
    for i in range(n):
        conv += [{"role": "agent", "content": "{...}"}, {"role": "monitor", "content": f"obs{i}"}]
    return conv


def test_window_keeps_only_last_k_observations():
    text = render_window(_conv(10), window=2)
    assert "obs9" in text and "obs8" in text and "obs7" not in text


def test_window_keeps_the_stated_task():
    assert "task: X" in render_window(_conv(10), window=2)


def test_window_excludes_agent_action_turns():
    assert "{...}" not in render_window(_conv(3), window=2)


def test_window_prompt_length_is_bounded_across_long_episode():
    assert len(render_window(_conv(40), window=2)) <= len(render_window(_conv(3), window=2)) + 50


def test_window_with_no_observations_yet():
    assert "task: X" in render_window(_conv(0), window=2)


def test_newest_observation_is_labelled():
    text = render_window(_conv(3), window=2)
    assert text.index("obs2") > text.index("newest")


def test_action_prompt_demands_single_json_last_and_shows_turns_left():
    p = action_user_prompt("SP", "W", turns_remaining=5)
    assert "JSON" in p and "5" in p and "SP" in p and "W" in p


def test_scratchpad_prompt_asks_for_rewrite():
    p = scratchpad_user_prompt("SP", "W", turns_remaining=5)
    assert "SP" in p and "W" in p and "scratchpad" in p.lower()


def test_action_handoff_asks_for_evidence_check_and_keeps_final_json():
    prompt = action_user_prompt("PLAN", "OBSERVATIONS", 5, json_only=True)
    assert "PLAN" in prompt and "OBSERVATIONS" in prompt
    assert "Carry out that plan unless" in prompt
    assert "prior submission feedback" in prompt
    assert "final answer must be exactly one JSON action" in prompt
