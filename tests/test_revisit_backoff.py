"""v6 revisit backoff: a task that stays dead on revisits is revisited less often, never banished."""
import json
import random

from redteamrl.train.allocation import AllocParams, TaskStats, allocate, revisit_gap
from redteamrl.train.banks import StartStateBank

P = AllocParams()
TASKS = [f"gen-{i}" for i in range(12)]


def _dead(task, since=0, dead_revisits=0):
    return TaskStats(task_id=task, history=[(0, 8)] * 3, iters_since_fresh=since,
                     dead_revisits=dead_revisits)


def _revisit_planned(since, dead_revisits):
    stats = [_dead(t) for t in TASKS]
    stats[0] = _dead("gen-0", since=since, dead_revisits=dead_revisits)
    banks = {t: StartStateBank(None) for t in TASKS}
    plans = allocate(stats, banks, 12, 3, random.Random(0), P)
    return any(p.kind == "fresh" and p.task_id == "gen-0" and p.reason == "revisit" for p in plans)


def test_gap_grows_then_holds_at_cap():
    gaps = [revisit_gap(_dead("t", dead_revisits=d), P) for d in (0, 1, 2, 3, 50)]
    assert gaps == [3, 6, 9, 9, 9]


def test_backed_off_task_is_not_due_until_its_gap():
    assert not _revisit_planned(since=5, dead_revisits=1)
    assert _revisit_planned(since=6, dead_revisits=1)


def test_never_banished():
    for d in (0, 1, 2, 3, 10, 100, 1000):
        assert _revisit_planned(since=9, dead_revisits=d), d


def test_note_fresh_outcome_rules():
    s = _dead("t")
    s.note_fresh_outcome(won_any=False, revisited=True)
    s.note_fresh_outcome(won_any=False, revisited=True)
    assert s.dead_revisits == 2
    s.note_fresh_outcome(won_any=False, revisited=False)
    assert s.dead_revisits == 2
    s.note_fresh_outcome(won_any=True, revisited=True)
    assert s.dead_revisits == 0


def test_stats_from_v5_dict_defaults_dead_revisits():
    s = TaskStats.from_dict({"task_id": "t", "history": [[0, 8]], "iters_since_fresh": 2})
    assert s.dead_revisits == 0
    s.dead_revisits = 2
    assert TaskStats.from_dict(s.to_dict()).dead_revisits == 2


def test_identity_round_trips_through_json():
    id_ = AllocParams().as_identity()
    assert json.loads(json.dumps(id_)) == id_
    assert id_["alloc_revisit_backoff"] == [3, 6, 9]
