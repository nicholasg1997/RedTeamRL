"""Sampling utilities for an informative, verdict-calibrated SFT mixture."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def _is_mandatory(example: dict) -> bool:
    return example.get("_source") != "authored"


def _deduplicate_easy_passes(examples: list[dict], cap: int) -> tuple[list[dict], int]:
    if cap < 1:
        raise ValueError("duplicate pass cap must be at least 1")
    seen: Counter[tuple[str, str]] = Counter()
    kept: list[dict] = []
    removed = 0
    for example in examples:
        if example.get("_class") != "pass" or _is_mandatory(example):
            kept.append(example)
            continue
        signature = (str(example.get("system", "")), str(example.get("prompt", "")))
        if seen[signature] >= cap:
            removed += 1
            continue
        seen[signature] += 1
        kept.append(example)
    return kept, removed


def _shuffle_copy(examples: list[dict], rng: Any) -> list[dict]:
    copied = list(examples)
    rng.shuffle(copied)
    # When a ratio clamp must discard examples, retain verified corrections before base-model
    # anchors. The shuffle still randomizes within each role.
    copied.sort(key=lambda example: example.get("_training_role") != "correction")
    return copied


def balance_sft_examples(
    examples: list[dict],
    rng: Any,
    *,
    max_call_pair_ratio: int = 2,
    duplicate_pass_cap: int = 2,
    redact_to_pass_ratio: float = 2.0,
) -> tuple[list[dict], dict]:
    """Return a balanced mixture and diagnostics.

    Call decisions retain the original bounded allow/reject ratio. Result decisions are skewed to
    ``redact_to_pass_ratio`` within every episode-type stratum after capping duplicate authored
    pass prompts. The default mirrors ``leak_penalty:over_refusal_penalty`` (2.0:1.0): the SFT
    objective scores a missed redaction and an unnecessary one identically, so the mixture is the
    only place that asymmetry can enter. The ratio is a ceiling, not a quota — a stratum with too
    few redactions still contributes all of them.

    Synthetic examples are mandatory curriculum: all are retained, and an imbalance in a synthetic
    stratum is rejected rather than silently dropping a carefully constructed counterexample. They
    are therefore exempt from the skew, which dilutes it; ``achieved_redact_share`` reports what the
    model actually sees.
    """
    if max_call_pair_ratio < 1:
        raise ValueError("max_call_pair_ratio must be at least 1")
    if redact_to_pass_ratio <= 0:
        raise ValueError("redact_to_pass_ratio must be positive")

    deduplicated, duplicate_passes_removed = _deduplicate_easy_passes(
        examples, duplicate_pass_cap
    )
    allows = _shuffle_copy(
        [example for example in deduplicated if example.get("_class") == "allow"], rng
    )
    rejects = _shuffle_copy(
        [example for example in deduplicated if example.get("_class") == "reject"], rng
    )
    call_limit = max_call_pair_ratio * min(len(allows), len(rejects))
    allows = allows[:call_limit]
    rejects = rejects[:call_limit]

    result_by_stratum: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"pass": [], "redact": []}
    )
    for example in deduplicated:
        verdict = example.get("_class")
        if verdict not in {"pass", "redact"}:
            continue
        stratum = str(example.get("_episode_type", "?"))
        result_by_stratum[stratum][verdict].append(example)

    passes: list[dict] = []
    redacts: list[dict] = []
    result_counts: dict[str, dict[str, int]] = {}
    for stratum in sorted(result_by_stratum):
        stratum_passes = _shuffle_copy(result_by_stratum[stratum]["pass"], rng)
        stratum_redacts = _shuffle_copy(result_by_stratum[stratum]["redact"], rng)
        mandatory_passes = [example for example in stratum_passes if _is_mandatory(example)]
        mandatory_redacts = [example for example in stratum_redacts if _is_mandatory(example)]
        if mandatory_passes or mandatory_redacts:
            if len(mandatory_passes) != len(mandatory_redacts):
                raise ValueError(
                    f"mandatory result curriculum is imbalanced in {stratum}: "
                    f"pass={len(mandatory_passes)} redact={len(mandatory_redacts)}"
                )
            # Synthetic curricula are already constructed as pairs; preserve every pair.
            selected_passes = stratum_passes
            selected_redacts = stratum_redacts
            if len(selected_passes) != len(selected_redacts):
                raise ValueError(
                    f"mandatory result stratum is not exactly paired in {stratum}: "
                    f"pass={len(selected_passes)} redact={len(selected_redacts)}"
                )
        else:
            # Take as many redactions as the pass supply can support at the target ratio, then
            # only the passes needed to stay under it. At ratio 1.0 this is min(passes, redacts).
            redact_target = min(
                len(stratum_redacts), math.floor(len(stratum_passes) * redact_to_pass_ratio)
            )
            pass_target = min(
                len(stratum_passes), math.ceil(redact_target / redact_to_pass_ratio)
            )
            selected_passes = stratum_passes[:pass_target]
            selected_redacts = stratum_redacts[:redact_target]
        passes.extend(selected_passes)
        redacts.extend(selected_redacts)
        # Exact 1:1 makes the kept counts symmetric, which hides which side the clamp bit.
        # Trimming surplus passes is the intended behaviour; culling redacts means the pass
        # supply — after deduplication — is what capped the redaction curriculum.
        result_counts[stratum] = {
            "pass": len(selected_passes),
            "redact": len(selected_redacts),
            "pass_dropped": len(stratum_passes) - len(selected_passes),
            "redact_dropped": len(stratum_redacts) - len(selected_redacts),
        }

    balanced = allows + rejects + passes + redacts
    rng.shuffle(balanced)
    result_total = len(passes) + len(redacts)
    selected_ids = {id(example) for example in balanced}
    corrections_available = [
        example
        for example in deduplicated
        if example.get("_training_role") == "correction"
    ]
    return balanced, {
        "duplicate_authored_passes_removed": duplicate_passes_removed,
        # Exempt synthetic pairs pull this below the target; it is the rate the model actually
        # sees, and therefore the rate a base-rate-matching defender would redact at.
        "achieved_redact_share": (len(redacts) / result_total) if result_total else 0.0,
        "target_redact_share": redact_to_pass_ratio / (1.0 + redact_to_pass_ratio),
        "allow": len(allows),
        "reject": len(rejects),
        "pass": len(passes),
        "redact": len(redacts),
        "corrections_available": len(corrections_available),
        "corrections_kept": sum(id(example) in selected_ids for example in corrections_available),
        "result_by_episode_type": result_counts,
    }
