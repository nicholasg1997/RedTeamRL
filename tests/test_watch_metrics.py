"""watch_metrics: pull each run's metrics.jsonl off the volume and serve the dashboard locally."""
import importlib.util
import json
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

_spec = importlib.util.spec_from_file_location("watch_metrics", "scripts/watch_metrics.py")
wm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wm)


def test_run_name_and_volume_path():
    assert wm.run_name("/runs/attacker-grpo-v6-arm2-coef0.0/") == "attacker-grpo-v6-arm2-coef0.0"
    assert wm.volume_path("/runs/defender-grpo-r9-vllm") == "defender-grpo-r9-vllm"
    assert wm.volume_path("already/relative") == "already/relative"


def test_sync_command_shape():
    assert wm.sync_command("redteamrl-eval-runs", "/runs/x", "/tmp/l/x/metrics.jsonl") == [
        "modal", "volume", "get", "--force", "redteamrl-eval-runs", "x/metrics.jsonl",
        "/tmp/l/x/metrics.jsonl"]


def test_write_manifest_roundtrip(tmp_path):
    wm.write_manifest(str(tmp_path), ["b", "a"])
    assert json.load(open(tmp_path / "runs.json")) == {"runs": ["b", "a"]}
    assert not list(tmp_path.glob("*.tmp"))


def test_parse_roots_names_and_duplicates():
    assert wm.parse_roots(["v6=/runs/a"], ["/runs/b"]) == {"v6": "/runs/a", "b": "/runs/b"}
    with pytest.raises(SystemExit):
        wm.parse_roots(["b=/runs/a"], ["/runs/b"])


@pytest.mark.parametrize("failure", ["exit1", "nomodal", "timeout"])
def test_sync_failure_does_not_raise(tmp_path, monkeypatch, failure):
    target = tmp_path / "run" / "metrics.jsonl"
    target.parent.mkdir()
    target.write_text("old\n")

    def fake_run(cmd, **kwargs):
        if failure == "exit1":
            return subprocess.CompletedProcess(cmd, 1, "", "not found")
        if failure == "nomodal":
            raise FileNotFoundError("modal")
        raise subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr(wm.subprocess, "run", fake_run)
    assert wm.sync_one("v", "/runs/run", str(target)) is False
    assert target.read_text() == "old\n"


def test_sync_success_replaces_atomically(tmp_path, monkeypatch):
    target = tmp_path / "run" / "metrics.jsonl"

    def fake_run(cmd, **kwargs):
        with open(cmd[-1], "w") as handle:
            handle.write('{"iter": 0}\n')
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(wm.subprocess, "run", fake_run)
    assert wm.sync_one("v", "/runs/run", str(target)) is True
    assert target.read_text() == '{"iter": 0}\n'
    assert not list(target.parent.glob("*.tmp"))


def test_server_serves_page_and_data(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<html>dash</html>")
    data = tmp_path / "data"
    data.mkdir()
    wm.write_manifest(str(data), ["a"])
    server = ThreadingHTTPServer(("127.0.0.1", 0), wm.make_handler(str(data), str(page)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as resp:
            assert resp.read() == b"<html>dash</html>"
            assert resp.headers["Cache-Control"] == "no-store"
        with urllib.request.urlopen(base + "/runs.json") as resp:
            assert json.loads(resp.read()) == {"runs": ["a"]}
            assert resp.headers["Cache-Control"] == "no-store"
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "/nope/metrics.jsonl")
        assert err.value.code == 404
    finally:
        server.shutdown()
