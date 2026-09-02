"""Convert collected records into grounded SFT examples.

Wins contribute all their turns; near-misses contribute only the pivotal turn;
others are discarded. Each authored turn is ground-truth checked before inclusion.
"""
from __future__ import annotations

from redteamrl.attack_sft.nearmiss import classify_episode, pivotal_turn
from redteamrl.attack_sft.rationalize import author_turn, WIN_FRAMING, NEARMISS_FRAMING
from redteamrl.attack_sft.example import build_example


def records_to_examples(
    records: list[dict],
    attacker_system: str,
    generate,
    include_near_misses: bool = True,
) -> list[dict]:
    """Convert collected episode records into grounded SFT examples.

    For each record:
    - Win: author all turns with WIN_FRAMING, include all that are grounded.
    - Near-miss: author only the pivotal turn with NEARMISS_FRAMING (ONLY when
      `include_near_misses` is True).
    - Other: skip entirely.

    WARNING on near-misses (`include_near_misses`): a near-miss's pivotal turn is the FAILING
    action -- a wrong submit, or the read the monitor redacted -- and this function keeps that
    action verbatim as the SFT target. Training on it teaches the model to reproduce its own
    failures; the first run that actually absorbed it (LR 1e-5) crashed held-out win rate 13.3%
    -> 2.3%. Using a near-miss correctly needs a CORRECT action (branch-and-verify, deferred),
    not the failing one. Callers should pass `include_near_misses=False` until that exists.

    Args:
        records: List of episode records, each with keys:
            - "canary": the secret string
            - "turns": list of turn dicts with keys:
                - "turn": turn index
                - "observable_prompt": what the attacker saw
                - "action": the action taken (dict)
                - "redaction_status": (for near_miss classification)
            - "won": bool (from ground-truth egress check)
        attacker_system: The system prompt for the attacker policy.
        generate: Callable (system_prompt, messages) -> str for re-authoring.

    Returns:
        Flat list of grounded examples: [{"system", "prompt", "target"}, ...].
    """
    examples = []

    for record in records:
        episode_type, _ = classify_episode(record)

        if episode_type == "win":
            # Contribute all turns
            for turn_dict in record["turns"]:
                reasoning = author_turn(
                    turn_dict["observable_prompt"],
                    turn_dict["action"],
                    WIN_FRAMING,
                    generate,
                )
                example = build_example(
                    attacker_system,
                    turn_dict["observable_prompt"],
                    reasoning,
                    turn_dict["action"],
                    record["canary"],
                )
                if example is not None:
                    examples.append(example)

        elif episode_type == "near_miss" and include_near_misses:
            # Contribute only the pivotal turn -- BUT this trains the FAILING action (see the
            # warning in the docstring). Gated off by default in the entrypoint.
            piv_turn_idx = pivotal_turn(record)
            if piv_turn_idx is not None:
                # Find the turn dict with this turn index
                turn_dict = None
                for t in record["turns"]:
                    if t["turn"] == piv_turn_idx:
                        turn_dict = t
                        break

                if turn_dict is not None:
                    reasoning = author_turn(
                        turn_dict["observable_prompt"],
                        turn_dict["action"],
                        NEARMISS_FRAMING,
                        generate,
                    )
                    example = build_example(
                        attacker_system,
                        turn_dict["observable_prompt"],
                        reasoning,
                        turn_dict["action"],
                        record["canary"],
                    )
                    if example is not None:
                        examples.append(example)

    return examples
