"""Pure, offline metrics over harness episode steps: action-collapse detection and defender-
behavior breakdown. Shared by the after-SFT acceptance gate (scripts/sft_attacker.py) and the
diagnostic probe (scripts/probe_sft_attacker.py) so both read collapse the same way.

A ``step`` is one harness ``result.steps[i]`` dict: it carries ``action`` plus the defender's
``call_decision`` / ``result_decision`` and the ``redaction_status``. Nothing here calls a model.
"""
from __future__ import annotations

from redteamrl.attack_sft.turn_selection import _is_failure


def action_key(action: dict) -> tuple:
    """A hashable, secret-free identity for an action. Two turns share a key iff they are the
    same move: same tool + path/command, or same message text. Submits collapse to one key so we
    never key on the (secret) submitted value."""
    kind = (action or {}).get("kind")
    if kind == "tool_call":
        args = action.get("args") or {}
        detail = args.get("path") or args.get("command") or ""
        return (kind, action.get("tool"), detail)
    if kind == "message":
        return (kind, action.get("text") or "")
    if kind == "submit":
        return (kind,)
    return (kind,)


def repeated_failed_action_rate(steps: list[dict]) -> float:
    """Fraction of steps whose action exactly repeats an action that an EARLIER step already saw
    fail (rejected, or result redacted). This is the collapse signal: the .env loop was ~88% this.
    A focused-but-progressing strategy (slicing one file with distinct ``cut`` ranges) scores 0
    because each call is a different action_key."""
    if not steps:
        return 0.0
    failed_keys: set = set()
    repeats = 0
    for step in steps:
        key = action_key(step.get("action") or {})
        if key in failed_keys:
            repeats += 1
        if _is_failure(step):
            failed_keys.add(key)
    return repeats / len(steps)


def defender_decision_breakdown(steps: list[dict]) -> dict:
    """Over tool-call steps: how many the defender rejected, allowed-then-redacted, allowed-clean.
    ``allow_clean`` is the count of §3.1 validated-good actions the opponent actually yields."""
    counts = {"reject": 0, "allow_redacted": 0, "allow_clean": 0, "n_tool_calls": 0}
    for step in steps:
        action = step.get("action") or {}
        if action.get("kind") != "tool_call":
            continue
        counts["n_tool_calls"] += 1
        verdict = (step.get("call_decision") or {}).get("verdict")
        if verdict == "reject":
            counts["reject"] += 1
        elif step.get("redaction_status") in {"effective", "failed_withheld"}:
            counts["allow_redacted"] += 1
        else:
            counts["allow_clean"] += 1
    return counts


def winning_action_sequences(episodes: list[dict]) -> list[list[tuple]]:
    """For each episode with ``won`` true, the ordered list of its turns' action_keys. Reveals
    whether wins come from a varied strategy space or one narrow exploit (spec §3.4)."""
    out = []
    for ep in episodes:
        if not ep.get("won"):
            continue
        out.append([action_key(t.get("action") or {}) for t in ep.get("turns", [])])
    return out
