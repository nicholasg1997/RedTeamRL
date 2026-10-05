from redteamrl.attack_sft.acceptance import accept

BASE = {"win_rate": 0.21, "effective_rank": 10.8}

def _after(win=0.21, rank=10.6, repeat=0.05):
    return {"win_rate": win, "effective_rank": rank, "repeated_no_progress_rate": repeat}

def test_promotes_when_held_flat_and_diverse():
    ok, reasons = accept(BASE, _after())          # win held, rank held, low repeat
    assert ok and reasons == []

def test_win_climb_is_NOT_required():
    ok, _ = accept(BASE, _after(win=0.21))        # exactly equal, no climb
    assert ok

def test_rejects_win_collapse():
    ok, reasons = accept(BASE, _after(win=0.023))  # the observed 2.3% collapse
    assert not ok and any("win" in r for r in reasons)

def test_rejects_action_collapse_even_if_win_and_rank_hold():
    # this is the case eff_rank MISSED: rank fine, win fine, but actions loop
    ok, reasons = accept(BASE, _after(repeat=0.60))
    assert not ok and any("repeat" in r.lower() for r in reasons)

def test_rejects_rank_collapse():
    ok, reasons = accept(BASE, _after(rank=5.0))
    assert not ok and any("rank" in r.lower() for r in reasons)


# --- revised gate (2026-10-02): relative loop limit + collapse-floor win ---

def test_moderate_win_dip_is_tolerated():
    # 24% -> 16% is a noise-level dip (not a halving); must still promote
    ok, reasons = accept({"win_rate": 0.24, "effective_rank": 10.8}, _after(win=0.16, rank=10.6))
    assert ok, reasons

def test_win_collapse_still_rejected_at_half():
    ok, reasons = accept({"win_rate": 0.24, "effective_rank": 10.8}, _after(win=0.08))
    assert not ok and any("win" in r for r in reasons)      # v2's 24->8 halving

def test_loop_limit_is_relative_to_base_when_provided():
    before = {"win_rate": 0.24, "effective_rank": 10.8, "repeated_no_progress_rate": 0.35}
    # after loop no worse than base (0.33 <= 0.35+slack) -> ok even though > absolute 0.25
    ok, reasons = accept(before, _after(win=0.20, repeat=0.33))
    assert ok, reasons
    # after loop clearly worse than base -> reject
    ok2, reasons2 = accept(before, _after(win=0.20, repeat=0.55))
    assert not ok2 and any("repeat" in r.lower() for r in reasons2)
