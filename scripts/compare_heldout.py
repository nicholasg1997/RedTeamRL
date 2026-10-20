"""Paired held-out comparison: is a checkpoint really better than the base policy?

    uv run python scripts/compare_heldout.py /runs/attacker-heldout-baseline/base-n128.json \\
        v6-it4=/runs/attacker-grpo-v6-arm2-coef0.0/heldout/iter4.json \\
        v6-it9=/runs/attacker-grpo-v6-arm2-coef0.0/heldout/iter9.json

The first argument is the reference (usually the base policy); each later one is LABEL=PATH or a
bare PATH. `/runs/...` paths are fetched from the Modal volume; anything else is read locally.
For each checkpoint it prints both win rates on the shared seeds, the discordant pairs (+ seeds the
checkpoint wins that the reference lost, - the reverse) and McNemar's exact two-sided p-value.
Read-only: nothing here writes to the volume.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redteamrl.eval.heldout_record import load_record, paired_compare  # noqa: E402

MOUNT = "/runs/"


def is_volume_path(path: str) -> bool:
    return path.startswith(MOUNT)


def fetch_command(volume: str, path: str, local_path: str) -> list[str]:
    return ["modal", "volume", "get", "--force", volume, path.removeprefix(MOUNT), local_path]


def _load(volume: str, path: str, tmp_dir: str) -> dict:
    if not is_volume_path(path):
        return load_record(path)
    local = os.path.join(tmp_dir, f"{len(os.listdir(tmp_dir))}.json")
    proc = subprocess.run(fetch_command(volume, path, local), capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(local):
        raise SystemExit(f"could not fetch {path} from volume {volume}: {proc.stderr.strip()}")
    return load_record(local)


def report(base: dict, checkpoints: dict[str, dict]) -> str:
    width = max([len(label) for label in checkpoints] + [10])
    lines = [f"{'checkpoint':<{width}}  {'n':>4}  win rate (ref -> ckpt)  discordant  p (McNemar)"]
    for label, record in checkpoints.items():
        c = paired_compare(base, record)
        lines.append(f"{label:<{width}}  {c['n']:>4}  "
                     f"{c['base_rate']:.1%} -> {c['new_rate']:.1%}".ljust(width + 30)
                     + f"  +{c['only_new']}/-{c['only_base']}".ljust(12)
                     + f"  p={c['p_value']:.3f}")
    lines.append("+ = seeds the checkpoint wins that the reference lost; - = the reverse. Seeds both "
                 "win or both lose carry no information about the difference.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", help="the reference record (usually the base policy)")
    parser.add_argument("checkpoints", nargs="+", help="LABEL=PATH or PATH, one per checkpoint")
    parser.add_argument("--volume", default="redteamrl-eval-runs")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        base = _load(args.volume, args.reference, tmp)
        checkpoints = {}
        for item in args.checkpoints:
            label, _, path = item.rpartition("=") if "=" in item else ("", "", item)
            checkpoints[label or path] = _load(args.volume, path, tmp)
    print(report(base, checkpoints))


if __name__ == "__main__":
    main()
