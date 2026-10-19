"""MetricsLog: one structured row per training iteration, never able to break the run."""
import json

from redteamrl.train.metrics_log import MetricsLog, read_metrics


def test_appends_one_row_per_call(tmp_path):
    path = str(tmp_path / "run" / "metrics.jsonl")
    log = MetricsLog(path)
    log.log(0, "attacker", cum_gen=100, **{"learning.fresh_win": 0.1})
    log.log(1, "attacker", cum_gen=250, **{"learning.fresh_win": 0.2})
    rows = read_metrics(path)
    assert [r["iter"] for r in rows] == [0, 1]
    assert rows[1]["phase"] == "attacker" and rows[1]["cum_gen"] == 250
    assert rows[0]["learning.fresh_win"] == 0.1
    assert isinstance(rows[0]["wall_time"], float)


def test_non_finite_values_become_null(tmp_path):
    path = str(tmp_path / "metrics.jsonl")
    MetricsLog(path).log(0, "attacker", **{"update.kl": float("nan"), "update.ratio": float("inf")})
    row = json.loads(open(path).read().strip())
    assert row["update.kl"] is None and row["update.ratio"] is None


def test_rejects_non_scalars_without_raising(tmp_path, capsys):
    path = str(tmp_path / "metrics.jsonl")
    MetricsLog(path).log(0, "attacker", bad=[1, 2], **{"learning.mean_reward": 0.5})
    row = read_metrics(path)[0]
    assert "bad" not in row and row["learning.mean_reward"] == 0.5
    assert "metrics" in capsys.readouterr().out


def test_unwritable_path_warns_and_does_not_raise(tmp_path, capsys):
    blocker = tmp_path / "file.txt"
    blocker.write_text("not a directory")
    MetricsLog(str(blocker / "metrics.jsonl")).log(0, "attacker", **{"update.kl": 0.1})
    assert "[metrics]" in capsys.readouterr().out


def test_read_skips_a_truncated_last_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"iter": 1, "phase": "attacker"}) + "\n" + '{"iter": 2, "le')
    assert len(read_metrics(str(path))) == 1
