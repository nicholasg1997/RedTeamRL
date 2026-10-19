"""Live training dashboard: pull each run's metrics.jsonl off the Modal volume and serve it locally.

    python scripts/watch_metrics.py /runs/attacker-grpo-v6-arm2-coef0.0
    python scripts/watch_metrics.py --root v5=/runs/attacker-grpo-v5-arm2-coef0.0 --root v6=/runs/...

Every --poll seconds each run's file is fetched with `modal volume get` (into a temp file, then moved
into place, so the page never reads a half-downloaded file) and served on localhost with the page in
scripts/dashboard/index.html. Read-only: nothing here writes to the volume or touches training.
--no-sync serves an existing directory as-is (e.g. the committed sample in scripts/dashboard/sample).
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "index.html")
MOUNT = "/runs/"          # where the training containers mount the runs volume


def run_name(ckpt_root: str) -> str:
    return os.path.basename(ckpt_root.rstrip("/"))


def volume_path(ckpt_root: str) -> str:
    """The CLI addresses the volume from its root; the drivers' roots are container paths."""
    return ckpt_root.removeprefix(MOUNT)


def sync_command(volume: str, root: str, local_path: str) -> list[str]:
    return ["modal", "volume", "get", "--force", volume,
            f"{volume_path(root).rstrip('/')}/metrics.jsonl", local_path]


def sync_one(volume: str, root: str, local_path: str, timeout: float = 60) -> bool:
    """Fetch one run's file. False (never an exception) when it is missing yet, the CLI is absent,
    or the call times out; the previous local copy is left as it was."""
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    tmp = local_path + ".tmp"
    try:
        proc = subprocess.run(sync_command(volume, root, tmp), capture_output=True, text=True,
                              timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if proc.returncode != 0 or not os.path.exists(tmp):
        return False
    os.replace(tmp, local_path)
    return True


def write_manifest(local_dir: str, names: list[str]) -> None:
    path = os.path.join(local_dir, "runs.json")
    with open(path + ".tmp", "w") as handle:
        json.dump({"runs": list(names)}, handle)
    os.replace(path + ".tmp", path)


def parse_roots(named: list[str], positional: list[str]) -> dict[str, str]:
    roots: dict[str, str] = {}
    pairs = [entry.split("=", 1) for entry in named] + [[run_name(r), r] for r in positional]
    for name, root in pairs:
        if name in roots:
            raise SystemExit(f"duplicate run name {name!r}; give one a --root NAME=ROOT")
        roots[name] = root
    return roots


def make_handler(local_dir: str, page: str):
    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store")   # every poll sees the newest file
            super().end_headers()

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html"):
                with open(page, "rb") as handle:
                    body = handle.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, *args):        # keep the terminal for sync status
            pass

    return functools.partial(Handler, directory=local_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("roots", nargs="*", help="checkpoint roots, e.g. /runs/attacker-grpo-v6-...")
    parser.add_argument("--root", action="append", default=[], help="NAME=ROOT (repeatable)")
    parser.add_argument("--volume", default="redteamrl-eval-runs")
    parser.add_argument("--poll", type=float, default=20.0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--no-sync", action="store_true", help="serve --local-dir as-is")
    args = parser.parse_args()

    local_dir = args.local_dir or tempfile.mkdtemp(prefix="redteamrl-metrics-")
    roots = parse_roots(args.root, args.roots)
    if not args.no_sync:
        if not roots:
            raise SystemExit("give at least one checkpoint root (or --no-sync with --local-dir)")
        os.makedirs(local_dir, exist_ok=True)
        write_manifest(local_dir, list(roots))

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(local_dir, PAGE))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"dashboard: http://localhost:{args.port}/   (serving {local_dir})", flush=True)
    try:
        while True:
            if not args.no_sync:
                for name, root in roots.items():
                    local = os.path.join(local_dir, name, "metrics.jsonl")
                    if not sync_one(args.volume, root, local):
                        print(f"[watch] {name}: no metrics.jsonl yet (will retry)", flush=True)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
