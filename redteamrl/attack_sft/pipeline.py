"""Convert collected records into grounded SFT examples via turn-level authoring (spec §3-§9).

Every turn of every collected episode -- win or loss -- is a candidate example. `classify_turn`
(turn_selection.py) decides per turn whether it trains and whether its action is masked. Boundary
turns are authored with a build-time teacher drawn from the nearest PRIOR observed failure --
never the turn's own outcome -- so the trained reasoning stays an inference from what the attacker
observed, not a quote of the monitor's private reasoning (spec §5). Each authored turn is
ground-truth checked (`build_example` / `is_grounded`) before inclusion.
"""
from __future__ import annotations

from redteamrl.attack_sft.turn_selection import classify_turn, _is_failure
from redteamrl.attack_sft.rationalize import author_turn, ROUTINE_FRAMING, BOUNDARY_FRAMING, WIN_FRAMING
from redteamrl.attack_sft.example import build_example

_FRAMINGS = {"routine": ROUTINE_FRAMING, "boundary": BOUNDARY_FRAMING, "win": WIN_FRAMING}


def _teacher_for(record: dict, i: int) -> str | None:
    """The nearest PRIOR turn's observed-failure reason, for authoring turn `i`'s boundary framing.

    Walks backwards from `i` (never including `i` itself -- a turn's own outcome is never its own
    lesson, per spec §4/§5) for the first `_is_failure` turn. A rejected call explains itself via
    `call_decision`; a call that was allowed but whose result was redacted explains itself via
    `result_decision` instead (its `call_decision` would just say "allow").
    """
    for j in range(i - 1, -1, -1):
        turn = record["turns"][j]
        if not _is_failure(turn):
            continue
        call = turn.get("call_decision") or {}
        if call.get("verdict") == "reject":
            return call.get("reasoning")
        result = turn.get("result_decision") or {}
        return result.get("reasoning")
    return None


def records_to_examples(records: list[dict], attacker_system: str, generate) -> list[dict]:
    """Convert collected episode records into grounded, turn-level SFT examples.

    For every turn of every record, `classify_turn` decides whether it contributes and whether
    its action is trained or masked (spec §4). Contributing turns are re-authored with the
    framing `classify_turn` selected; boundary turns additionally get a build-time teacher (the
    nearest prior observed failure's actual reason) to make the inferred boundary accurate,
    without that private reasoning ever entering the SFT prompt itself (only `observable_prompt`
    does -- see `build_example`).

    Args:
        records: List of episode records, each with keys:
            - "canary": the secret string
            - "won": bool (from ground-truth egress check)
            - "turns": list of turn dicts -- see turn_selection.classify_turn for the fields read
              (observable_prompt, action, call_decision, result_decision, redaction_status).
        attacker_system: The system prompt for the attacker policy.
        generate: Callable (system_prompt, messages) -> str for re-authoring.

    Returns:
        Flat list of grounded examples: [{"system", "prompt", "target", "action_loss_weight"}, ...].
    """
    examples = []

    for record in records:
        for i, turn in enumerate(record["turns"]):
            selection = classify_turn(record, i)
            if not selection["train"]:
                continue

            framing = _FRAMINGS[selection["framing"]]
            teacher = _teacher_for(record, i) if selection["framing"] == "boundary" else None

            reasoning = author_turn(
                turn["observable_prompt"],
                turn["action"],
                framing,
                generate,
                teacher=teacher,
            )
            example = build_example(
                attacker_system,
                turn["observable_prompt"],
                reasoning,
                turn["action"],
                record["canary"],
                train_action=selection["train_action"],
            )
            if example is not None:
                examples.append(example)

    return examples
