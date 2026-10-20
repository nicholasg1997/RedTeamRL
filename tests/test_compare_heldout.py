from redteamrl.eval.heldout_record import make_record
from scripts.compare_heldout import fetch_command, is_volume_path, report


def _rec(outcomes):
    return make_record(list(range(1000, 1000 + len(outcomes))), outcomes)


def test_volume_paths_are_recognised_and_fetched_from_the_volume_root():
    assert is_volume_path("/runs/attacker-heldout-baseline/base-n128.json")
    assert not is_volume_path("eval-runs/base-n128.json")
    assert fetch_command("redteamrl-eval-runs", "/runs/a/heldout/iter4.json", "/tmp/x.json") == [
        "modal", "volume", "get", "--force", "redteamrl-eval-runs", "a/heldout/iter4.json",
        "/tmp/x.json"]


def test_report_shows_rates_discordant_pairs_and_p_value_per_checkpoint():
    base = _rec(["timeout"] * 8 + ["leak", "leak"])
    out = report(base, {"v6 it4": _rec(["leak"] * 7 + ["timeout"] * 2 + ["leak"]),
                        "v6 it9": _rec(["timeout"] * 10)})
    lines = out.splitlines()
    it4 = next(l for l in lines if l.startswith("v6 it4"))
    assert "20.0% -> 80.0%" in it4 and "+7/-1" in it4 and "p=0.070" in it4
    it9 = next(l for l in lines if l.startswith("v6 it9"))
    assert "+0/-2" in it9
