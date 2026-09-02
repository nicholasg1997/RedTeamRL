from redteamrl.attack_sft.pipeline import records_to_examples

def _rec(**k): return {"canary":"sk-CANARY-x", "turns":[], **k}

def test_win_contributes_all_turns_nearmiss_only_pivotal():
    win = _rec(won=True, turns=[
        {"turn":0,"observable_prompt":"a","action":{"kind":"message","text":"x"}},
        {"turn":1,"observable_prompt":"b","action":{"kind":"message","text":"y"}}])
    nm = _rec(won=False, turns=[
        {"turn":0,"observable_prompt":"c","action":{"kind":"tool_call"},"redaction_status":None},
        {"turn":1,"observable_prompt":"d","action":{"kind":"submit","value":"WRONG"},"redaction_status":None}])
    exs = records_to_examples([win, nm], "atk-sys", lambda s, m: "grounded reason")
    # 2 win turns + 1 nearmiss pivotal turn = 3
    assert len(exs) == 3

def test_other_contributes_nothing():
    other = _rec(won=False, turns=[{"turn":0,"observable_prompt":"c","action":{"kind":"tool_call"},"redaction_status":None}])
    assert records_to_examples([other], "atk-sys", lambda s, m: "r") == []


def test_near_misses_excluded_when_flag_false():
    """include_near_misses=False must yield ONLY win examples -- a near-miss's pivotal action is
    the FAILING move (regression: training on it crashed win rate 13.3% -> 2.3%)."""
    from redteamrl.attack_sft.pipeline import records_to_examples
    win = {"canary": "sk-CANARY-x", "won": True, "turns": [
        {"turn": 0, "observable_prompt": "a", "action": {"kind": "message", "text": "x"}}]}
    near = {"canary": "sk-CANARY-x", "won": False, "turns": [
        {"turn": 0, "observable_prompt": "c", "action": {"kind": "submit", "value": "WRONG"},
         "redaction_status": None}]}
    both = records_to_examples([win, near], "sys", lambda s, m: "grounded reason",
                               include_near_misses=False)
    only_win = records_to_examples([win], "sys", lambda s, m: "grounded reason",
                                   include_near_misses=False)
    assert len(both) == len(only_win) == 1     # the near-miss contributed nothing
