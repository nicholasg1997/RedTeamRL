"""Per-turn selection for reflective attacker SFT (spec §4). Pure -- no model, no judge.

Train the REASONING on every turn that has something learnable. Train the ACTION only when the
attacker got it right; mask it when it failed. A turn's reasoning reflects on PRIOR observed
failures, never its own outcome, so a rejected turn with no prior failure has nothing to teach.
"""
from __future__ import annotations

_SKIP = {"train": False, "train_action": False, "framing": None}


def _delivered_canary(turn: dict, canary: str) -> bool:
    return canary in (turn.get("returned") or "")


def _is_failure(turn: dict) -> bool:
    """A failure the attacker OBSERVED this turn: a rejected call, or a redaction that stripped
    the payload it would otherwise have received."""
    call = turn.get("call_decision") or {}
    if call.get("verdict") == "reject":
        return True
    if turn.get("redaction_status") in {"effective", "failed_withheld"}:
        return True
    return False


def _prior_failure(record: dict, i: int) -> bool:
    return any(_is_failure(record["turns"][j]) for j in range(i))


def classify_turn(record: dict, i: int) -> dict:
    turn = record["turns"][i]
    action = turn.get("action") or {}
    kind = action.get("kind")
    canary = record["canary"]

    if kind == "submit":
        if canary in (action.get("value") or ""):
            return {"train": True, "train_action": True, "framing": "win"}
        return dict(_SKIP)                                 # wrong submit: no observable signal

    prior = _prior_failure(record, i)

    if kind == "tool_call":
        verdict = (turn.get("call_decision") or {}).get("verdict", "reject")
        if verdict == "allow":
            framing = "win" if _delivered_canary(turn, canary) else ("boundary" if prior else "routine")
            return {"train": True, "train_action": True, "framing": framing}
        # rejected call: action masked; only worth training if there is a prior failure to reflect on
        if prior:
            return {"train": True, "train_action": False, "framing": "boundary"}
        return dict(_SKIP)

    if kind == "message":
        return {"train": True, "train_action": True, "framing": "boundary" if prior else "routine"}

    return dict(_SKIP)
