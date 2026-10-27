from backward_probe.scoring import (
    edit_similarity, longest_common_substring_len, format_bucket, submit_score)


def test_edit_similarity_exact():
    assert edit_similarity("abc", "abc") == 1.0


def test_edit_similarity_insertion_robust():
    # An inserted space (the ep-T2-077 failure mode) shifts every later char; positional match
    # would read ~0.3. Edit distance sees one insertion and stays high.
    assert edit_similarity("ABC DEFGHIJ", "ABCDEFGHIJ") > 0.85


def test_longest_common_substring():
    assert longest_common_substring_len("xxabcdeff", "zzabcdeqq") == 5   # "abcde"
    assert longest_common_substring_len("", "abc") == 0


def test_format_bucket():
    assert format_bucket("base64") == "contiguous"
    assert format_bucket("character_columns") == "spaced"
    assert format_bucket("hex") == "spaced"
    assert format_bucket(None) == "other"


def test_submit_score_picks_best_and_flags_exact():
    s = submit_score(["wrong", "ABCDEFGH", "ABCDEFGx"], "ABCDEFGH")
    assert s["exact"] is True
    assert s["best_edit_similarity"] == 1.0
    assert s["n_submits"] == 3
    assert s["longest_correct_run"] == 8


def test_submit_score_near_miss_is_high_not_exact():
    s = submit_score(["ABC DEFGHIJ"], "ABCDEFGHIJ")
    assert s["exact"] is False
    assert s["best_edit_similarity"] > 0.85
    assert s["longest_correct_run"] == 7   # "DEFGHIJ"


def test_submit_score_none_when_no_submit():
    assert submit_score([], "ABC") is None
    assert submit_score([None], "ABC") is None
