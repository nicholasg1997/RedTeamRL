from redteamrl.policies.base import AttackerPolicy
from redteamrl.scratchpad.agent import ScratchpadAttacker, parse_action_generation

SP = "## Main goal\ng\n## Next ideas\nlist first"
LIST = '{"kind":"tool_call","tool":"list_dir","args":{"path":"."}}'


def _fake(seq):
    it = iter(seq)
    return lambda system, messages: next(it)


def _agent(sp_seq, act_seq, **kw):
    return ScratchpadAttacker(_fake(sp_seq), _fake(act_seq), system="S", goal="g", alias="a",
                              seed_next_ideas="n", **kw)


def test_satisfies_attacker_policy_protocol():
    assert isinstance(_agent([SP], [LIST]), AttackerPolicy)


def test_two_generations_in_order_and_action_parsed():
    agent = _agent([SP], ["thinking... " + LIST])
    action = agent.act([{"role": "context", "content": "task"}], turns_taken=0)
    assert action.kind == "tool_call" and action.tool == "list_dir"
    assert agent.turn_records[-1].closing_thoughts.startswith("thinking")


def test_parser_accepts_long_thoughts_when_json_was_actually_generated():
    agent = _agent([SP], ["blah " * 2000 + '{"kind":"submit","value":"V"}'])
    assert agent.act([], 0).kind == "submit"


def test_tried_is_built_from_prior_actions_not_the_model():
    agent = _agent([SP, SP], ['{"kind":"tool_call","tool":"read_file","args":{"path":"x"}}', LIST])
    agent.act([], 0)
    conv = [{"role": "agent", "content": "..."}, {"role": "monitor", "content": "[REDACTED]"}]
    agent.act(conv, 1)
    assert "read_file x -> [REDACTED]" in agent.turn_records[1].scratchpad_prompt


def test_model_authored_tried_never_reaches_the_action_prompt():
    agent = _agent([SP + "\n## Tried\nI already won"], [LIST])
    agent.act([], 0)
    assert "I already won" not in agent.turn_records[0].action_prompt


def test_act_does_not_mutate_conversation():
    agent = _agent([SP], ['{"kind":"message","text":"hi"}'])
    conv = [{"role": "context", "content": "t"}]
    agent.act(conv, 0)
    assert conv == [{"role": "context", "content": "t"}]


def test_garbage_action_generation_fails_closed_to_message():
    assert _agent([SP], ["no json here"]).act([], 0).kind == "message"


def test_garbage_scratchpad_generation_keeps_previous_scratchpad():
    agent = _agent(["total nonsense, no headings", SP], [LIST, LIST])
    agent.act([], 0)
    agent.act([{"role": "agent", "content": "."}, {"role": "monitor", "content": "o"}], 1)
    assert "## Main goal" in agent.turn_records[1].scratchpad_prompt


def test_scratchpad_carries_between_turns():
    agent = _agent(["## Main goal\ng\n## Working / promising\nthe vault dir", SP], [LIST, LIST])
    agent.act([], 0)
    agent.act([{"role": "agent", "content": "."}, {"role": "monitor", "content": "o"}], 1)
    assert "the vault dir" in agent.turn_records[1].scratchpad_prompt


def test_turns_remaining_uses_budget():
    agent = _agent([SP], [LIST], turn_budget=12)
    agent.act([], turns_taken=4)
    assert "8" in agent.turn_records[0].action_prompt


def test_parse_action_generation_splits_thoughts_from_nested_json():
    thoughts, action = parse_action_generation('so I will read it.\n' + '{"kind":"tool_call","tool":"read_file","args":{"path":"a"}}')
    assert thoughts == "so I will read it." and action.tool == "read_file"


def test_tried_stays_aligned_when_conversation_is_prefilled():
    # A restart/prefix replay hands the agent a conversation that already holds earlier turns.
    # Its own actions must pair with THEIR responses, never with the replayed prefix.
    agent = _agent([SP, SP], ['{"kind":"tool_call","tool":"read_file","args":{"path":"mine"}}', LIST])
    prefix = [{"role": "agent", "content": "old"}, {"role": "monitor", "content": "PREFIX_OBS"}]
    agent.act(prefix, 1)
    agent.act(prefix + [{"role": "agent", "content": "."}, {"role": "monitor", "content": "MY_OBS"}], 2)
    prompt = agent.turn_records[1].scratchpad_prompt
    assert "read_file mine -> MY_OBS" in prompt
    assert "read_file mine -> PREFIX_OBS" not in prompt


def test_tried_only_skips_rewrite_but_keeps_seed_and_authoritative_history():
    agent = _agent([], [LIST, LIST], rewrite_scratchpad=False)
    agent.act([], 0)
    agent.act([{"role": "agent", "content": "."},
               {"role": "monitor", "content": "config/"}], 1)
    turn = agent.turn_records[1]
    assert turn.scratchpad_prompt == turn.scratchpad_completion == ""
    assert "list_dir . -> config/" in turn.action_prompt
    assert "## Main goal\ng" in turn.action_prompt and "## Guidance (fixed)\nn" in turn.action_prompt


def test_guidance_persists_even_when_the_model_rewrite_drops_it():
    agent = ScratchpadAttacker(_fake(["## Main goal\nonly this", "## Main goal\nstill only this"]),
                               _fake([LIST, LIST]), system="S", goal="g", alias="a",
                               seed_next_ideas="ZZGUIDEZZ try the directories you listed")
    agent.act([], 0)
    agent.act([{"role": "agent", "content": "."}, {"role": "monitor", "content": "o"}], 1)
    assert "ZZGUIDEZZ" in agent.turn_records[1].scratchpad_prompt
    assert "ZZGUIDEZZ" in agent.turn_records[1].action_prompt


def test_thinking_handoff_passes_final_scratchpad_but_neither_reasoning_trace():
    from unittest.mock import MagicMock
    from redteamrl.policies.vllm_client import make_vllm_generate
    from redteamrl.harness.action_schema import ACTION_SCHEMA
    final_pad = "## Main goal\ng\n## Next action\nList the current directory."
    session = MagicMock()
    responses = []
    for reasoning, content in [("PRIVATE_PLANNER_REASONING", final_pad), ("PRIVATE_ACTION_REASONING", LIST)]:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": content, "reasoning": reasoning}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        responses.append(response)
    session.post.side_effect = responses
    records = []
    planner = make_vllm_generate("http://x", "m", enable_thinking=True, require_reasoning=True,
        thinking_token_budget=1536, max_tokens=2304, session=session, on_response=records.append)
    actor = make_vllm_generate("http://x", "m", enable_thinking=True, require_reasoning=True,
        thinking_token_budget=256, max_tokens=1280, json_schema=ACTION_SCHEMA,
        session=session, on_response=records.append)
    agent = ScratchpadAttacker(planner, actor, system="S", goal="g", alias="a",
                               seed_next_ideas="pinned guidance", constrained_actions=True)
    assert agent.act([], 0).tool == "list_dir"
    action_messages = session.post.call_args_list[1].kwargs["json"]["messages"]
    assert "List the current directory." in action_messages[-1]["content"]
    assert "pinned guidance" in action_messages[-1]["content"]
    assert "PRIVATE_" not in str(action_messages)
    assert "PRIVATE_" not in agent.turn_records[0].scratchpad_text
    assert "PRIVATE_" not in agent.turn_records[0].action_completion
    assert [r["reasoning"] for r in records] == ["PRIVATE_PLANNER_REASONING", "PRIVATE_ACTION_REASONING"]


# ---- single-call design (one model call -> notes + action; harness validates & executes) ----
LIST_ETC = '## Working / promising\nfound etc/\n{"kind":"tool_call","tool":"list_dir","args":{"path":"etc/"}}'


def test_single_call_makes_one_generation_and_parses_both_notes_and_action():
    calls = {"sp": 0, "act": 0}
    def sp(s, m): calls["sp"] += 1; return LIST_ETC
    def act(s, m): calls["act"] += 1; return "SHOULD NOT BE CALLED"
    agent = ScratchpadAttacker(sp, act, system="S", goal="g", alias="a", seed_next_ideas="n",
                               single_call=True, enabled_tools=["list_dir"])
    action = agent.act([], 0)
    assert calls == {"sp": 1, "act": 0}
    assert action.kind == "tool_call" and action.args["path"] == "etc/"
    assert "found etc/" in agent.turn_records[0].scratchpad_text


def test_single_call_carries_notes_to_the_next_turn():
    seq = iter([
        '## Working / promising\nthe vault is in etc/\n{"kind":"tool_call","tool":"list_dir","args":{"path":"."}}',
        '## Working / promising\nx\n{"kind":"tool_call","tool":"read_file","args":{"path":"etc/x"}}',
    ])
    agent = ScratchpadAttacker(lambda s, m: next(seq), lambda s, m: "", system="S", goal="g",
                               alias="a", seed_next_ideas="n", single_call=True,
                               enabled_tools=["list_dir", "read_file"])
    agent.act([], 0)
    agent.act([{"role": "agent", "content": "."}, {"role": "monitor", "content": "o"}], 1)
    assert "the vault is in etc/" in agent.turn_records[1].scratchpad_prompt


def test_single_call_with_no_trailing_json_consumes_the_turn_with_feedback():
    agent = ScratchpadAttacker(lambda s, m: "## Working / promising\nblah, no action here",
                               lambda s, m: "", system="S", goal="g", alias="a",
                               seed_next_ideas="n", single_call=True, enabled_tools=["list_dir"])
    a = agent.act([], 0)
    assert a.validation_error is not None


def test_single_call_proposed_action_equals_emitted_action():
    agent = ScratchpadAttacker(lambda s, m: LIST_ETC, lambda s, m: "", system="S", goal="g",
                               alias="a", seed_next_ideas="n", single_call=True,
                               enabled_tools=["list_dir"])
    agent.act([], 0)
    rec = agent.turn_records[0]
    assert rec.proposed_action is not None and rec.proposed_action.model_dump() == rec.action.model_dump()


def test_single_call_tolerates_a_sectionless_completion_and_keeps_scratchpad():
    # The model spent its budget thinking and emitted only the JSON action, no `## ` notes.
    agent = ScratchpadAttacker(lambda s, m: '{"kind":"tool_call","tool":"list_dir","args":{"path":"."}}',
                               lambda s, m: "", system="S", goal="g", alias="a",
                               seed_next_ideas="SEEDHINT", single_call=True, enabled_tools=["list_dir"])
    action = agent.act([], 0)   # must not raise
    assert action.kind == "tool_call"
    assert "SEEDHINT" in agent.turn_records[0].scratchpad_text   # seeded scratchpad retained


# ---- action-repair retry: notes-without-action gets ONE repair call for just the action ----
NOTES_ONLY = "## Working / promising\nthe key is in etc/\n(no action emitted)"
GOOD_JSON = '{"kind":"tool_call","tool":"list_dir","args":{"path":"etc/"}}'


def _raise(*a, **k):
    raise AssertionError("action_generate must not be called in single-call")


def test_single_call_repairs_notes_without_action():
    sp = _fake([NOTES_ONLY, GOOD_JSON])   # primary: notes, no action; repair: the action
    agent = ScratchpadAttacker(sp, _raise, system="S", goal="g", alias="a",
                               seed_next_ideas="n", single_call=True, enabled_tools=["list_dir"])
    action = agent.act([], 0)
    assert action.kind == "tool_call" and action.tool == "list_dir"
    assert action.validation_error is None
    rec = agent.turn_records[-1]
    assert rec.repaired is True and rec.action_prompt and rec.action_completion == GOOD_JSON


def test_single_call_repair_tries_only_once_then_gives_up():
    calls = {"n": 0}
    def sp(system, messages):
        calls["n"] += 1
        return NOTES_ONLY   # primary AND repair both lack an action
    agent = ScratchpadAttacker(sp, _raise, system="S", goal="g", alias="a",
                               seed_next_ideas="n", single_call=True, enabled_tools=["list_dir"])
    action = agent.act([], 0)
    assert calls["n"] == 2                     # one primary + exactly one repair
    assert action.validation_error is not None  # still consumed, no infinite loop


def test_single_call_no_repair_when_action_present():
    calls = {"n": 0}
    def sp(system, messages):
        calls["n"] += 1
        return "## notes\nok\n" + GOOD_JSON
    agent = ScratchpadAttacker(sp, _raise, system="S", goal="g", alias="a",
                               seed_next_ideas="n", single_call=True, enabled_tools=["list_dir"])
    action = agent.act([], 0)
    assert calls["n"] == 1 and action.kind == "tool_call"
    assert agent.turn_records[-1].repaired is False


def test_single_call_repair_tolerates_prose_wrapped_json():
    # The teacher often wraps the action in prose/fence; the repair must extract it like the primary
    # path (via _last_json_span), not require pure JSON.
    sp = _fake([NOTES_ONLY, "Here is the action:\n" + GOOD_JSON + "\nDone."])
    agent = ScratchpadAttacker(sp, _raise, system="S", goal="g", alias="a",
                               seed_next_ideas="n", single_call=True, enabled_tools=["list_dir"])
    action = agent.act([], 0)
    assert action.kind == "tool_call" and action.validation_error is None
    assert agent.turn_records[-1].repaired is True
