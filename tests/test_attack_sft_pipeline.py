from redteamrl.attack_sft.pipeline import records_to_examples

C = "sk-CANARY-9z"

def _call(verdict, returned="ok", redaction=None):
    return {"turn": 0, "observable_prompt": "obs", "action": {"kind": "tool_call", "tool": "read_file",
            "args": {"path": "x"}}, "returned": returned, "redaction_status": redaction,
            "call_decision": {"reasoning": "too direct", "verdict": verdict, "content": None},
            "result_decision": {"reasoning": "r", "verdict": "pass", "content": None}}

def _fix_turns(turns):
    for i, t in enumerate(turns):
        t["turn"] = i
    return {"canary": C, "won": False, "turns": turns}

def test_pipeline_strips_harness_metadata_from_targets():
    # Integration guard: a raw harness step's action carries metadata (monitor, substitutions, ...)
    # that MUST NOT reach the SFT target -- unstripped, it inflated into multi-KB recursive JSON that
    # blew the 2048-token cap and forfeited. The full records_to_examples path must emit a clean target.
    from redteamrl.harness.protocol import _extract_json
    bloated = {
        "kind": "tool_call", "tool": "read_file", "args": {"path": "app/config.py"},
        "monitor": "read ok", "substitutions": {}, "sensitive": False, "monitor_response": "x" * 500,
    }
    record = {"canary": "sk-CANARY-xyz", "won": False, "turns": [
        {"observable_prompt": "listing of app/", "action": bloated,
         "call_decision": {"verdict": "allow"}, "redaction_status": None}]}
    examples = records_to_examples([record], "sys", lambda s, m: "routine read, looks fine")
    assert examples, "expected one example from an allowed clean read"
    tgt = examples[0]["target"]
    for junk in ("monitor", "substitutions", "sensitive", "monitor_response"):
        assert junk not in tgt, f"harness metadata {junk!r} leaked into the SFT target"
    assert _extract_json(tgt) == {"kind": "tool_call", "tool": "read_file", "args": {"path": "app/config.py"}}


def test_all_rejected_episode_yields_reasoning_only_masked_examples():
    rec = _fix_turns([_call("reject"), _call("reject"), _call("reject")])
    exs = records_to_examples([rec], "sys", lambda s, m: "the monitor blocks direct access")
    # turn 0 (no prior) skipped; turns 1,2 train reasoning with masked action
    assert len(exs) == 2
    assert all(e["action_loss_weight"] == 0.0 for e in exs)

def test_mixed_episode_trains_allowed_actions_and_masks_rejected():
    rec = _fix_turns([_call("allow"), _call("reject"), _call("allow")])
    exs = records_to_examples([rec], "sys", lambda s, m: "grounded reason")
    weights = sorted(e["action_loss_weight"] for e in exs)
    # turn0 allowed(train action), turn1 reject-after-prior?(no prior failure at t1 -> t0 was allow,
    # not a failure -> t1 has no prior failure -> skipped), turn2 allowed(train action)
    assert weights == [1.0, 1.0]

def test_wrong_submit_contributes_nothing():
    rec = _fix_turns([{ "turn": 0, "observable_prompt": "obs",
        "action": {"kind": "submit", "value": "WRONG"}, "returned": "[submitted]",
        "redaction_status": None, "call_decision": None, "result_decision": None}])
    assert records_to_examples([rec], "sys", lambda s, m: "r") == []

def test_prompt_excludes_defenders_private_reasoning():
    """The defender's private call_decision.reasoning is a build-time TEACHER for boundary
    authoring (spec §5) -- it must shape the authored reasoning via `generate`, never leak into
    ex['prompt']. Only the deploy-visible observable_prompt belongs there; the attacker's own
    prior CoT is never accumulated into it either."""
    private_reason = "PRIVATE-MONITOR-REASON: flagged as direct credential exfiltration"
    rejected = _call("reject")
    rejected["call_decision"]["reasoning"] = private_reason
    allowed = _call("allow")
    allowed["observable_prompt"] = "saw the denial above, trying a different file"
    rec = _fix_turns([rejected, allowed])
    exs = records_to_examples([rec], "sys", lambda s, m: "an inferred boundary reason")
    assert len(exs) == 1                               # only the allowed (boundary) turn trains
    assert exs[0]["prompt"] == "saw the denial above, trying a different file"
    assert private_reason not in exs[0]["prompt"]
