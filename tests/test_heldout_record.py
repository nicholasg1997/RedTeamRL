import json
import math

import pytest

from redteamrl.eval.heldout_record import (load_record, make_record, mcnemar_exact_p,
                                           paired_compare, save_record)


def _rec(outcomes, start=1000, **meta):
    seeds = list(range(start, start + len(outcomes)))
    return make_record(seeds, outcomes, turns=[3] * len(outcomes), **meta)


def test_make_record_counts_leaks_as_wins_and_keeps_meta():
    rec = _rec(["leak", "timeout", "leak", "timeout"], iter=4)
    assert rec["n"] == 4 and rec["wins"] == 2 and rec["win_rate"] == 0.5
    assert rec["seeds"] == [1000, 1001, 1002, 1003]
    assert rec["outcomes"] == ["leak", "timeout", "leak", "timeout"]
    assert rec["iter"] == 4


def test_make_record_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        make_record([1000, 1001], ["leak"])


def test_save_then_load_round_trips(tmp_path):
    rec = _rec(["leak", "timeout"])
    path = tmp_path / "heldout" / "iter4.json"
    save_record(str(path), rec)
    assert load_record(str(path)) == rec
    assert not (tmp_path / "heldout" / "iter4.json.tmp").exists()


def test_mcnemar_exact_matches_binomial_tail():
    # 8 discordant pairs split 7/1: two-sided exact p = 2 * P(X <= 1 | n=8, p=.5) = 2 * 9/256.
    assert math.isclose(mcnemar_exact_p(7, 1), 18 / 256)
    assert mcnemar_exact_p(1, 7) == mcnemar_exact_p(7, 1)
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(4, 4) == 1.0


def test_paired_compare_counts_discordant_pairs():
    base = _rec(["leak", "timeout", "timeout", "timeout", "leak"])
    new = _rec(["leak", "leak", "leak", "timeout", "timeout"])
    cmp = paired_compare(base, new)
    assert cmp["n"] == 5
    assert (cmp["both"], cmp["only_new"], cmp["only_base"], cmp["neither"]) == (1, 2, 1, 1)
    assert cmp["base_rate"] == 0.4 and cmp["new_rate"] == 0.6
    assert cmp["p_value"] == mcnemar_exact_p(2, 1)


def test_paired_compare_uses_only_shared_seeds():
    # A 128-seed baseline still pairs with a 256-seed eval: only seeds present in both count.
    base = _rec(["timeout"] * 4, start=1000)
    new = _rec(["leak"] * 6, start=1002)
    cmp = paired_compare(base, new)
    assert cmp["n"] == 2 and cmp["only_new"] == 2


def test_paired_compare_refuses_disjoint_records():
    with pytest.raises(ValueError):
        paired_compare(_rec(["leak"], start=1000), _rec(["leak"], start=2000))


def test_record_is_plain_json():
    json.dumps(_rec(["leak", "timeout"], iter=9))


def test_load_record_expands_the_first_baseline_runs_seed_range(tmp_path):
    # The first --baseline-eval launch (before records existed) wrote "seeds": [start, end).
    path = tmp_path / "base-n4.json"
    path.write_text(json.dumps({"n": 4, "wins": 1, "win_rate": 0.25, "seeds": [1000, 1004],
                                "outcomes": ["leak", "timeout", "timeout", "timeout"]}))
    rec = load_record(str(path))
    assert rec["seeds"] == [1000, 1001, 1002, 1003]
    assert paired_compare(rec, _rec(["leak", "leak", "timeout", "timeout"]))["only_new"] == 1
