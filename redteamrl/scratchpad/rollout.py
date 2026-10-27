"""Run one scratchpad-attacker episode and bundle its private trajectory with the result.

The harness loop is untouched: the agent is just an `AttackerPolicy`. The per-turn scratchpad
trajectory is read back from `agent.turn_records` afterwards and kept as a research artifact
(and, later, as the (prompt, completion) pairs RL scores).
"""
import json
import os
from dataclasses import asdict, dataclass

from redteamrl.harness.episode import EpisodeResult, run_episode
from redteamrl.scratchpad.agent import ScratchpadAttacker, ScratchpadTurn


@dataclass
class ScratchpadRolloutResult:
    result: EpisodeResult
    turns: list[ScratchpadTurn]
    tier: str


def run_scratchpad_rollout(spec, agent: ScratchpadAttacker, defender, sandbox, *, max_turns: int,
                           tier: str, on_turn=None) -> ScratchpadRolloutResult:
    result = run_episode(spec, agent, defender, sandbox, max_turns=max_turns, on_turn=on_turn)
    return ScratchpadRolloutResult(result=result, turns=list(agent.turn_records), tier=tier)


def turn_to_jsonable(t: ScratchpadTurn) -> dict:
    """One turn as JSON-safe data. The single place AttackerAction fields are converted, so the
    completed and failed-episode record paths cannot drift (a drift crashed single-call-v1)."""
    d = asdict(t)
    d["action"] = t.action.model_dump()
    d["proposed_action"] = t.proposed_action.model_dump() if t.proposed_action else None
    return d


def to_jsonable(r: ScratchpadRolloutResult) -> dict:
    return {"tier": r.tier, "result": r.result.model_dump(),
            "turns": [turn_to_jsonable(t) for t in r.turns]}


def save_rollout(path: str, r: ScratchpadRolloutResult) -> None:
    """Atomic write, so a reader never sees a half-written record."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        json.dump(to_jsonable(r), handle, indent=2)
    os.replace(path + ".tmp", path)
