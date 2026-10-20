"""v7 finalize: early (same-iteration) banking is idempotent; restart entries that come back all-win
at their floor graduate; same-iteration replacements are counted."""
from test_v6_finalize import IT, N, P, _entry, _fresh, _group, _restart, _setup

from redteamrl.train.v5_finalize import bank_episodes
from redteamrl.train.v6_finalize import v6_identity
from redteamrl.train.v7_finalize import finalize_v7, v7_identity


def _run(episodes, base, reps, banks, stats, rec, graduated=frozenset()):
    return finalize_v7(episodes, base, reps, banks, stats, {}, set(graduated), IT, N, rec, P)


def test_bank_episodes_is_idempotent(tmp_path):
    banks, _, rec = _setup(tmp_path)
    episodes = _group(_fresh(0), [True, False, False, False])
    assert bank_episodes(episodes, banks, IT, rec) == (1, 0)
    first = banks["A"].entries()
    assert bank_episodes(episodes, banks, IT, rec) == (1, 0)
    assert banks["A"].entries() == first


def test_all_win_restart_at_floor_graduates(tmp_path):
    entry = _entry("floor", key_turn=1)
    banks, stats, rec = _setup(tmp_path, entries=[entry])
    plan = _restart(0, entry, k=1, frac=1.0)
    summary, _, graduated = _run(_group(plan, [True] * N), [plan], [], banks, stats, rec)
    assert "floor" in graduated and summary["graduated"] == 1 and summary["graduated_total"] == 1


def test_all_win_restart_with_room_does_not_graduate(tmp_path):
    entry = _entry("room", key_turn=10)
    banks, stats, rec = _setup(tmp_path, entries=[entry])
    plan = _restart(0, entry, k=10, frac=1.0)
    summary, _, graduated = _run(_group(plan, [True] * N), [plan], [], banks, stats, rec)
    assert "room" not in graduated and summary["graduated"] == 0


def test_mixed_restart_at_floor_does_not_graduate(tmp_path):
    entry = _entry("floor", key_turn=1)
    banks, stats, rec = _setup(tmp_path, entries=[entry])
    plan = _restart(0, entry, k=1, frac=1.0)
    _, _, graduated = _run(_group(plan, [True, False, True, True]), [plan], [], banks, stats, rec)
    assert "floor" not in graduated


def test_graduated_ids_evicted_from_the_bank_are_dropped(tmp_path):
    entry = _entry("kept")
    banks, stats, rec = _setup(tmp_path, entries=[entry])
    plan = _fresh(0)
    _, _, graduated = _run(_group(plan, [False] * N), [plan], [], banks, stats, rec,
                           graduated={"kept", "gone"})
    assert graduated == {"kept"}


def test_same_iteration_replacements_are_counted(tmp_path):
    same = _entry("same", iteration=IT)
    older = _entry("older", iteration=IT - 1)
    banks, stats, rec = _setup(tmp_path, entries=[same, older])
    base = _fresh(0)
    rep_same = _restart(1, same, k=5, frac=1.0)
    rep_old = _restart(2, older, k=5, frac=1.0)
    episodes = {**_group(base, [False] * N), **_group(rep_same, [True, False, False, False]),
                **_group(rep_old, [False] * N)}
    summary, _, _ = _run(episodes, [base], [rep_same, rep_old], banks, stats, rec)
    assert summary["same_iter_replacements"] == 1 and summary["same_iter_replacements_live"] == 1


def test_v7_identity_extends_v6():
    ident = v7_identity(P)
    assert {k: v for k, v in ident.items() if k in v6_identity(P)} == v6_identity(P)
    assert ident["graduation"] == "all-win-at-floor-v1" and ident["same_iter_banking"] == 1


# --- v7.2: near misses restart AFTER the redaction ---------------------------------------------

from dataclasses import replace as _replace  # noqa: E402

from test_v6_finalize import _record  # noqa: E402

V72 = _replace(P, near_miss_offset=1, near_miss_min_turns_left=3)


def _near_miss(key_turn, n_steps=12, eid=0):
    return {eid: {"episode_id": eid, "task_id": "A", "kind": "fresh", "slot": 0, "won": False,
                  "reward": -0.1, "group_key": "A#slot0", "closeness": [1, key_turn],
                  "record": _record("nm", n_steps=n_steps), "restart_fallback": False}}


def test_near_miss_is_banked_at_the_turn_after_the_redaction(tmp_path):
    # Restarting AT the redaction turn re-reads the secret and is redacted again (v7/v7.1: 0/9 live).
    banks, _, rec = _setup(tmp_path)
    assert bank_episodes(_near_miss(4), banks, IT, rec, params=V72) == (0, 1)
    assert banks["A"].entries()[0].key_turn == 5


def test_near_miss_without_enough_turns_left_is_not_banked(tmp_path):
    banks, _, rec = _setup(tmp_path)
    assert bank_episodes(_near_miss(9), banks, IT, rec, params=V72) == (0, 0)  # 12 - 10 = 2 < 3
    assert bank_episodes(_near_miss(8), banks, IT, rec, params=V72) == (0, 1)  # 12 - 9 = 3


def test_wins_keep_their_decisive_turn(tmp_path):
    banks, _, rec = _setup(tmp_path)
    bank_episodes(_group(_fresh(0), [True, False, False, False]), banks, IT, rec, params=V72)
    assert banks["A"].entries()[0].key_turn == 4


def test_default_params_bank_near_misses_unchanged(tmp_path):
    banks, _, rec = _setup(tmp_path)
    bank_episodes(_near_miss(4), banks, IT, rec)
    assert banks["A"].entries()[0].key_turn == 4
