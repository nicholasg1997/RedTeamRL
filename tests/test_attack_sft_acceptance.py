from redteamrl.attack_sft.acceptance import accept


def test_promote_when_win_up_and_rank_preserved():
    ok, reasons = accept({"win_rate": 0.12, "effective_rank": 100.0}, {"win_rate": 0.20, "effective_rank": 98.0})
    assert ok and reasons == []


def test_veto_on_entropy_collapse_even_if_win_up():
    ok, reasons = accept({"win_rate": 0.12, "effective_rank": 100.0}, {"win_rate": 0.40, "effective_rank": 50.0})
    assert not ok and any("rank" in r for r in reasons)


def test_veto_when_win_not_up():
    ok, reasons = accept({"win_rate": 0.12, "effective_rank": 100.0}, {"win_rate": 0.12, "effective_rank": 100.0})
    assert not ok and any("win_rate" in r for r in reasons)
