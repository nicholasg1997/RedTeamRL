"""Restartable episodes: begin a new episode part-way through a recorded one.

A restart keeps the first ``k`` turns of a SOURCE episode as context and lets the current policy act
from turn ``k``. The prefix is rebuilt, never regenerated:

  * the transcript and the attacker's own output history come straight from the recorded strings, so
    the attacker's next prompt is exactly the one it saw originally;
  * the sandbox is rebuilt by re-running the source's *allowed* tool calls, then CHECKED against a
    content hash recorded live after that turn. A mismatch means the source is not reproducible
    (a command with time-varying output, say) and raises ``NondeterministicSource``: a state is
    never restored approximately;
  * defender decisions are taken from the record and never re-queried. The bookkeeping (counters,
    taint, completion) goes through ``finish_turn``, the same code a live turn uses, so a restored
    state cannot drift from a live one.

Selection of WHICH episodes to restart from lives in train/closeness.py and train/banks.py; none of
it touches reward, which stays the ground-truth egress outcome over the whole episode.
"""
from __future__ import annotations

import hashlib
import json
import os

from redteamrl.harness.episode import (EpisodeResult, RedactionEnforcement, TurnState,
                                       build_result, execute_tool, finish_turn, run_turns)
from redteamrl.types import AttackerAction, DefenderDecision, TaskSpec

RECORD_VERSION = 1


class NondeterministicSource(RuntimeError):
    """Re-running a source's prefix did not reproduce the sandbox it recorded."""


def sandbox_content_hash(root) -> str:
    """sha256 over the sandbox tree: directories, symlinks (by target text, never followed) and
    file bytes, in sorted order. Modification times and the root's own absolute path are excluded,
    so two sandboxes holding the same files hash equal."""
    root = str(root)
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, root)
        for name in sorted(dirnames + filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.normpath(os.path.join(rel_dir, name))
            if os.path.islink(full):
                digest.update(f"L:{rel}:{os.readlink(full)}\0".encode())
            elif os.path.isdir(full):
                digest.update(f"D:{rel}\0".encode())
            else:
                with open(full, "rb") as handle:
                    file_hash = hashlib.sha256(handle.read()).hexdigest()
                digest.update(f"F:{rel}:{file_hash}\0".encode())
    return digest.hexdigest()


class FsHashRecorder:
    """An ``on_turn`` hook that records the sandbox hash after every turn. It must run DURING the
    episode: the filesystem after turn t is gone once the episode ends."""

    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.hashes: list[str] = []

    def __call__(self, state: TurnState, turn: int) -> None:
        self.hashes.append(sandbox_content_hash(self.sandbox.root))


def record_episode(result: EpisodeResult, spec: TaskSpec, raw_outputs: list[str],
                   fs_hash_after: list[str], max_turns: int,
                   post_completion_turns: int | None = None) -> dict:
    """Everything needed to restart from this episode, JSON-serialisable. ``raw_outputs`` is the
    attacker's ``_outputs`` read after the episode; ``fs_hash_after`` is ``FsHashRecorder.hashes``."""
    return {
        "version": RECORD_VERSION,
        "spec": spec.model_dump(mode="json"),
        "steps": result.steps,
        "raw_outputs": list(raw_outputs),
        "fs_hash_after": list(fs_hash_after),
        "redaction_enforcement": result.redaction_enforcement,
        "max_turns": max_turns,
        "post_completion_turns": post_completion_turns,
        "outcome": result.outcome,
        "leak_at_turn": result.leak_at_turn,
    }


def save_record(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        json.dump(record, handle)
    os.replace(path + ".tmp", path)


def load_record(path: str) -> dict:
    with open(path) as handle:
        record = json.load(handle)
    if record.get("version") != RECORD_VERSION:
        raise ValueError(f"unsupported restart record version {record.get('version')!r}")
    return record


def _decision(raw: dict | None) -> DefenderDecision | None:
    return DefenderDecision.model_validate(raw) if raw else None


def _count_redaction(state: TurnState, result: DefenderDecision | None,
                     redaction_status: str | None) -> None:
    """The redaction counters a live turn increments while classifying the defender's result,
    reconstructed from the recorded verdict and ``redaction_status``."""
    if result is None:
        return
    state.result_decisions += 1
    if redaction_status in {"effective", "failed_withheld", "failed_delivered", "sensitive_pass"}:
        state.redaction_opportunities += 1
    if result.verdict == "redact":
        state.redaction_attempts += 1
    if redaction_status == "effective":
        state.effective_redactions += 1
    elif redaction_status in {"failed_withheld", "failed_delivered"}:
        state.failed_redactions += 1
    elif redaction_status == "unnecessary":
        state.unnecessary_redactions += 1
    elif redaction_status == "sensitive_pass":
        state.sensitive_result_passes += 1


def restore_state(record: dict, k: int, attacker, sandbox) -> TurnState:
    """Rebuild the state after ``k`` prefix turns (``1 <= k <= len(steps)``).

    Resets ``sandbox`` to the SOURCE's task (its spec and canary come from the record, never
    regenerated), re-runs the allowed tool calls, verifies the filesystem hash, and sets the
    attacker's output history to what it held after turn ``k - 1``."""
    spec = TaskSpec.model_validate(record["spec"])
    steps = record["steps"]
    if not 1 <= k <= len(steps):
        raise ValueError(f"restart turn k={k} outside 1..{len(steps)}")
    if getattr(attacker, "history_turns", 0) > 0 and len(record["raw_outputs"]) < k:
        raise ValueError(f"the attacker keeps output history but the record holds only "
                         f"{len(record['raw_outputs'])} outputs for k={k}")

    sandbox.reset(spec)
    state = TurnState.start(spec)
    post_completion_turns = record["post_completion_turns"]
    for step in steps[:k]:
        action = AttackerAction.model_validate(step["action"])
        call_decision = _decision(step["call_decision"])
        result_decision = _decision(step["result_decision"])
        if (action.kind == "tool_call" and action.tool in spec.enabled_tools
                and call_decision is not None and call_decision.verdict == "allow"):
            execute_tool(action, sandbox)
        _count_redaction(state, result_decision, step["redaction_status"])
        was_complete = state.complete
        stopped = finish_turn(state, spec, step["turn"], step["phase"], was_complete, action,
                              call_decision, result_decision, step["policy_returned"],
                              step["returned"], step["redaction_status"], post_completion_turns)
        if stopped:
            raise ValueError(f"restart turn k={k} is at or past the source's stopping turn "
                             f"({step['turn']}); a restart must stop before it")
    recorded = record["fs_hash_after"][k - 1]
    if sandbox_content_hash(sandbox.root) != recorded:
        raise NondeterministicSource(
            f"re-running the first {k} turns did not reproduce the recorded sandbox")
    if hasattr(attacker, "_outputs"):
        attacker._outputs = list(record["raw_outputs"][:k])
    return state


def run_restart(record: dict, k: int, attacker, defender, sandbox, max_turns: int,
                post_completion_turns: int | None,
                redaction_enforcement: RedactionEnforcement, on_turn=None) -> EpisodeResult:
    """Restore the source at turn ``k`` and let ``attacker`` / ``defender`` play on. The result
    covers the WHOLE episode (prefix included), so reward is the whole-episode reward; turn
    indexes stay absolute and the turn limit is not reset."""
    if (record["redaction_enforcement"] != redaction_enforcement
            or record["max_turns"] != max_turns
            or record["post_completion_turns"] != post_completion_turns):
        raise ValueError("restart settings differ from the source episode's "
                         "(redaction_enforcement / max_turns / post_completion_turns)")
    spec = TaskSpec.model_validate(record["spec"])
    state = restore_state(record, k, attacker, sandbox)
    run_turns(state, spec, attacker, defender, sandbox, max_turns, post_completion_turns,
              redaction_enforcement, start_turn=k, on_turn=on_turn)
    return build_result(state, spec, max_turns, redaction_enforcement)
