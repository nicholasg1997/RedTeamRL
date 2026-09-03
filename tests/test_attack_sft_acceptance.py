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
