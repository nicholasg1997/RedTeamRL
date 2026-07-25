"""Build a conservative correction-and-anchor curriculum from banked decisions."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

from redteamrl.harness.protocol import _DEFENDER_FAIL_MARKER
from redteamrl.harness.taint import sensitive_spans
from redteamrl.sft.attribute import correct_verdict
from redteamrl.sft.example import render_target, strip_to_sft
from redteamrl.sft.filter import is_grounded, redaction_effective, redaction_ok


def base_decision_is_correct(record: dict) -> bool:
    """Whether the deployed base monitor already made the mechanically verified decision.

    Anchors are trained on VERBATIM, so they must clear the SAME bars as a rationalized example:
      * grounded reasoning -- a correct verdict justified by the hidden role teaches the monitor to
        reason from information it will not have at deployment;
      * effective (not minimal) redaction -- gating on exact spans here while the rest of the
        pipeline accepts any confined redaction that clears the taint would relabel working
        decisions as mistakes and spend best-of-k re-authoring them.
    """
    correct = correct_verdict(record)
    decision = record.get("decision") or {}
    if correct is None or decision.get("verdict") != correct:
        return False
    reasoning = str(decision.get("reasoning", ""))
    if reasoning.startswith(_DEFENDER_FAIL_MARKER):
        return False
    if not is_grounded(
        reasoning,
        record.get("forbidden") or [],
        record.get("observable_prompt") or "",
        record.get("required") or [],
        authoring_values=sensitive_spans(
            record.get("raw_result") or "", record.get("forbidden") or []
        ),
    ):
        return False
    if correct == "redact":
        return redaction_effective(
            decision.get("remove") or [],
            record.get("raw_result") or "",
            record.get("forbidden") or [],
            record.get("required") or [],
        )
    return True


def _canonicalize_target(example: dict) -> dict:
    """Convert legacy prose-plus-JSON targets into the current single JSON object."""
    normalized = dict(example)
    raw_target = str(normalized["target"])
    try:
        trace = json.loads(raw_target.splitlines()[-1])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("accepted SFT example has no parseable final JSON target") from exc
    normalized["target"] = render_target(trace)
    return normalized


def build_curriculum_example(record: dict, rationalized_row: dict) -> dict | None:
    """Use correct base outputs as anchors and verified rewrites only to correct mistakes."""
    correct = correct_verdict(record)
    if correct is None:
        return None

    if base_decision_is_correct(record):
        decision = record["decision"]
        trace = {
            "reasoning": decision["reasoning"],
            "verdict": decision["verdict"],
            "remove": decision.get("remove"),
        }
        example = strip_to_sft(record, trace)
        example["_exact_span"] = (
            redaction_ok(decision.get("remove") or [], record.get("raw_result") or "",
                         record.get("forbidden") or [], record.get("required") or [])
            if correct == "redact" else None
        )
        role = "anchor"
    else:
        if not rationalized_row.get("accepted") or not rationalized_row.get("example"):
            return None
        example = _canonicalize_target(rationalized_row["example"])
        role = "correction"

    example["_class"] = correct
    example["_record_id"] = record["record_id"]
    example["_source"] = "authored"
    example["_episode_type"] = record["episode_type"]
    example["_training_role"] = role
    return example


def select_corrections_and_anchors(
    examples: list[dict],
    rng: Any,
    *,
    anchor_to_correction_ratio: float = 3.0,
    min_anchors_per_stratum: int = 4,
) -> tuple[list[dict], dict]:
    """Keep every correction and only enough correct base outputs to prevent drift.

    Strata are decision class x episode type. A stratum with no observed mistakes still retains a
    small anchor floor; a mistake-heavy stratum gets proportionally more anchors.
    """
    if anchor_to_correction_ratio < 0:
        raise ValueError("anchor_to_correction_ratio must be non-negative")
    if min_anchors_per_stratum < 0:
        raise ValueError("min_anchors_per_stratum must be non-negative")

    strata: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: {"anchor": [], "correction": []}
    )
    passthrough: list[dict] = []
    for example in examples:
        role = example.get("_training_role")
        if role not in {"anchor", "correction"}:
            passthrough.append(example)
            continue
        key = (str(example.get("_class", "?")), str(example.get("_episode_type", "?")))
        strata[key][role].append(example)

    selected = list(passthrough)
    by_stratum = {}
    for key in sorted(strata):
        anchors = list(strata[key]["anchor"])
        corrections = list(strata[key]["correction"])
        rng.shuffle(anchors)
        anchor_limit = max(
            min_anchors_per_stratum,
            math.ceil(len(corrections) * anchor_to_correction_ratio),
        )
        kept_anchors = anchors[:anchor_limit]
        selected.extend(corrections)
        selected.extend(kept_anchors)
        by_stratum[f"{key[0]}:{key[1]}"] = {
            "corrections": len(corrections),
            "anchors_available": len(anchors),
            "anchors_kept": len(kept_anchors),
            "anchors_dropped": len(anchors) - len(kept_anchors),
        }

    rng.shuffle(selected)
    return selected, {
        "corrections": sum(e.get("_training_role") == "correction" for e in selected),
        "anchors": sum(e.get("_training_role") == "anchor" for e in selected),
        "by_stratum": by_stratum,
    }
