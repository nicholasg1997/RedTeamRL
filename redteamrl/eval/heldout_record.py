"""Per-seed held-out records and the paired comparison they make possible.

The held-out eval is close to deterministic per seed: fixed seeds, a greedy attacker and a greedy
(temp-0) defender. So two policies evaluated on the same seeds are best compared PAIRED -- "which
seeds does the new policy win that the base lost, and vice versa?" -- with McNemar's exact test on
the discordant pairs, not by comparing two win rates as if they were independent samples. Seeds both
policies win (or both lose) carry no information about the difference, so the test spends its power
only where the policies actually disagree.

Only seeds present in BOTH records are compared, so a 128-seed baseline still pairs with the first
128 seeds of a larger eval.
"""
from __future__ import annotations

import json
import os
from math import comb

WIN = "leak"


def make_record(seeds: list[int], outcomes: list[str], **meta) -> dict:
	"""A JSON-safe held-out record. `meta` (iter, turns, config...) is stored alongside."""
	if len(seeds) != len(outcomes):
		raise ValueError(f"{len(seeds)} seeds but {len(outcomes)} outcomes")
	wins = sum(o == WIN for o in outcomes)
	return {"n": len(outcomes), "wins": wins,
	        "win_rate": wins / len(outcomes) if outcomes else 0.0,
	        "seeds": list(seeds), "outcomes": list(outcomes), **meta}


def save_record(path: str, record: dict) -> None:
	"""Atomic write: a reader never sees a half-written record."""
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path + ".tmp", "w") as handle:
		json.dump(record, handle, indent=2)
	os.replace(path + ".tmp", path)


def load_record(path: str) -> dict:
	with open(path) as handle:
		record = json.load(handle)
	seeds, outcomes = record["seeds"], record["outcomes"]
	if len(seeds) == 2 and len(outcomes) == seeds[1] - seeds[0] != 2:
		# The first --baseline-eval launch stored the seed RANGE [start, end), not the list.
		record["seeds"] = list(range(seeds[0], seeds[1]))
	return record


def mcnemar_exact_p(only_new: int, only_base: int) -> float:
	"""Two-sided exact McNemar p-value: under 'no difference' each discordant pair is a fair coin."""
	n = only_new + only_base
	if n == 0:
		return 1.0
	k = min(only_new, only_base)
	tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
	return min(1.0, 2 * tail)


def paired_compare(base: dict, new: dict) -> dict:
	"""Compare two records on their shared seeds. Raises if they share none."""
	base_won = {s: o == WIN for s, o in zip(base["seeds"], base["outcomes"])}
	new_won = {s: o == WIN for s, o in zip(new["seeds"], new["outcomes"])}
	shared = sorted(base_won.keys() & new_won.keys())
	if not shared:
		raise ValueError("the two records share no seeds; they cannot be paired")
	both = sum(base_won[s] and new_won[s] for s in shared)
	only_new = sum(new_won[s] and not base_won[s] for s in shared)
	only_base = sum(base_won[s] and not new_won[s] for s in shared)
	n = len(shared)
	return {"n": n, "both": both, "only_new": only_new, "only_base": only_base,
	        "neither": n - both - only_new - only_base,
	        "base_rate": (both + only_base) / n, "new_rate": (both + only_new) / n,
	        "p_value": mcnemar_exact_p(only_new, only_base),
	        "new_seeds": [s for s in shared if new_won[s] and not base_won[s]],
	        "lost_seeds": [s for s in shared if base_won[s] and not new_won[s]]}
