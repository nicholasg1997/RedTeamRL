"""Per-iteration training metrics as structured JSONL: the source of truth for the dashboard.

One JSON object per iteration, appended to `<ckpt_root>/metrics.jsonl` on the runs volume. Metric
keys are `<category>.<name>` (e.g. `learning.fresh_win`); the prefix IS the category, so adding a
metric is one key at the training site and the dashboard picks it up with no other change.

Never raises into training. A metrics failure costs one row of telemetry; letting it propagate would
cost the run. Every failure is printed as a `[metrics]` warning instead.
"""
from __future__ import annotations

import json
import math
import os
import time

_SCALARS = (bool, int, float, type(None))


def _clean(name: str, value):
    """A JSON-safe scalar, or a sentinel meaning "drop this key". NaN/inf become null so one bad
    value cannot make the whole line unparseable."""
    if not isinstance(value, _SCALARS):
        print(f"[metrics] dropping non-scalar {name!r} ({type(value).__name__})", flush=True)
        return _DROP
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


_DROP = object()


class MetricsLog:
    def __init__(self, path: str):
        self.path = path

    def log(self, iter: int, phase: str, cum_gen: int | None = None, **scalars) -> None:
        try:
            row = {"iter": iter, "cum_gen": cum_gen, "wall_time": time.time(), "phase": phase}
            for name, value in scalars.items():
                cleaned = _clean(name, value)
                if cleaned is not _DROP:
                    row[name] = cleaned
            line = json.dumps(row) + "\n"
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as handle:
                handle.write(line)            # one write per row: a reader sees whole lines or none
                handle.flush()
        except Exception as exc:              # never let telemetry kill a run
            print(f"[metrics] could not write iter {iter} to {self.path}: {exc!r}", flush=True)


def read_metrics(path: str) -> list[dict]:
    """Every parseable row. A torn or partial line (a sync that caught a write mid-flight) is
    skipped rather than failing the whole file."""
    rows = []
    try:
        with open(path) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows
