"""Assert the training config matches the one the viability gate actually measured.

The gate exists to produce a number training can rely on -- "16.7% win rate, TRAINABLE". That
number only transfers if training runs what was measured. It did not: TEMP (1.0 vs 0.7),
USE_PROMPT_POOL (rotating vs fixed defender prompt) and ATK_MAX_TOKENS each diverged, and
iteration 0 came back with a 0% win rate and 100% dead groups -- an iteration that produced no
gradient at all, against a measured rate whose 1-in-48 outcome had probability 0.0016.

None of those were visible until the rollout numbers arrived roughly an hour in. This module makes
the divergence fail at startup instead, and makes a DELIBERATE divergence a one-line explicit act
rather than an accident.

The measured config lives here, in version control, rather than in the volume file the gate writes:
it is then reviewable in a diff, survives a volume wipe, and does not force a gate re-run before
every training run. Pure functions -- no torch, no modal, no network.
"""

# What attacker_viability::check measured on 2026-08-31. The arm that passed and that
# train_attacker.py is meant to reproduce.
MEASURED_GATE_CONFIG = {
    "arm": "no-thinking-2048",
    "attacker_model": "Qwen/Qwen3-8B",
    "defender_model": "Qwen/Qwen3-4B",
    "defender_sft_adapter": "/runs/sft-defender/round9/training/7eeae96976a4/adapter",
    "defender_grpo_adapter": "/runs/defender-grpo-r9-vllm/iter1",
    "atk_temperature": 0.7,
    "atk_max_tokens": 2048,
    "atk_enable_thinking": False,
    "max_turns": 12,
    "redaction_enforcement": "unshielded",
    "canary_revision": 101,
    "defender_prompt": "fixed",     # "fixed" (DEFENDER_SYSTEM) | "pool" (DEFENDER_STYLES)
    "measured_win_rate": 0.167,     # for the error message: what you are giving up by diverging
}

_SKIP = {"arm", "measured_win_rate"}


def compare_configs(measured: dict, training: dict, acknowledged=()) -> list[str]:
    """One human-readable line per key that differs and was not explicitly acknowledged.

    A key the gate never recorded is reported too. Adding a knob should force a decision about
    whether it was validated, rather than silently passing as "no mismatch".
    """
    acknowledged = set(acknowledged)
    lines = []
    for key in sorted(set(measured) | set(training)):
        if key in acknowledged or key in _SKIP:
            continue
        if key not in measured:
            lines.append(f"  {key}: training={training[key]!r} but NOT RECORDED by the gate "
                         f"-- this knob has never been validated")
        elif key not in training:
            lines.append(f"  {key}: gate measured {measured[key]!r} but training does not set it")
        elif measured[key] != training[key]:
            lines.append(f"  {key}: gate measured {measured[key]!r}, training uses "
                         f"{training[key]!r}")
    return lines


def assert_matches_gate(measured: dict, training: dict, acknowledged=()) -> None:
    """Raise listing every unacknowledged mismatch. Called before any model loads."""
    mismatches = compare_configs(measured, training, acknowledged)
    if not mismatches:
        return
    rate = measured.get("measured_win_rate")
    context = (f" The gate measured a {rate:.1%} win rate under ITS config; a divergent run is not "
               f"entitled to that number." if rate is not None else "")
    raise RuntimeError(
        "Training config does not match the viability gate." + context + "\n"
        + "\n".join(mismatches)
        + "\n\nIf a difference is deliberate, add its key to ACKNOWLEDGED_CONFIG_DELTAS in the "
          "training script -- and expect the gate's win rate not to hold."
    )
