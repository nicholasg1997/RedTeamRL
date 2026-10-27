"""Pre-decision checkpoints and exact T2 prefix replay.

Harvests lack the filesystem hashes required by harness.restart.restore_state. Never invent
those hashes: replay through the actual turn gate, compare observations/decisions, then record
a workspace hash for subsequent restores. No recorded teacher commands run on the host.
"""
from dataclasses import asdict
import copy

from redteamrl.harness.episode import TurnState, finish_turn, step_turn
from redteamrl.harness.restart import _count_redaction
from redteamrl.scratchpad.document import HEADER, render_scratchpad, seed_scratchpad
from redteamrl.scratchpad.poc import SEED_NEXT_IDEAS
from redteamrl.scratchpad.render import action_contract, render_window, single_call_user_prompt
from redteamrl.scratchpad.tried import tried_lines
from redteamrl.types import AttackerAction, DefenderDecision, TaskSpec
from redteamrl.verify.defenders import OracleRedactorDefender
from teacher_sft.data import decoded_views, digest, split_for, witness_for


class ReplayMismatch(ValueError):
    pass


def state_digest(state):
    fields = asdict(state)
    fields["taint_seen"] = sorted(fields["taint_seen"])
    return digest(fields)


def decision(raw):
    return DefenderDecision(**raw) if raw else None


def recorded_state(record, k):
    """Reconstruct the recorded visible transcript/bookkeeping without executing commands."""
    spec = TaskSpec(**record["task"])
    state = TurnState.start(spec)
    for s in record["result"]["steps"][:k]:
        _count_redaction(state, decision(s["result_decision"]), s.get("redaction_status"))
        stop = finish_turn(state, spec, s["turn"], s["phase"], state.complete,
            AttackerAction(**s["action"]), decision(s["call_decision"]), decision(s["result_decision"]),
            s.get("policy_returned"), s["returned"], s.get("redaction_status"), None,
            execution=s.get("execution"))
        if stop:
            raise ValueError("Restart prefix already terminated")
    return state


def predecision_scratchpad(record, k, max_turns):
    """Read the pad from the saved INPUT, never the chosen turn's completion/reasoning."""
    conversation = recorded_state(record, k).conversation
    window = render_window(conversation, 2)
    prompt = record["turns"][k]["scratchpad_prompt"]
    suffix = single_call_user_prompt("", window, max_turns - k) + "\n\n" + action_contract(
        record["task"]["enabled_tools"], with_notes=True)
    if not prompt.endswith(suffix):
        raise ValueError("Unknown saved prompt format; cannot extract pre-decision scratchpad")
    pad = prompt[:-len(suffix)]
    if not pad.startswith(HEADER):
        raise ValueError("Missing saved scratchpad header")
    return pad


def reconstruction_witness(conversation, secret):
    # Only returned observations, never teacher claims or oracle-containing metadata.
    observations = [(i, m) for i, m in enumerate(conversation) if m["role"] in {"monitor", "harness"}]
    # Match the student's actual two-observation window, not inaccessible older evidence.
    for i, message in observations[-2:]:
        for method, text in decoded_views(message["content"]):
            if secret in text:
                return dict(message_index=i, method=method,
                            observation_sha256=digest(message["content"]))
    return None


def make_cases(record, max_turns, max_depth=3):
    steps = record["result"]["steps"]
    length = len(steps)
    secret = record["task"]["forbidden"]
    if len(secret) != 1:
        raise ValueError("Diagnostic requires exactly one synthetic canary")
    secret = secret[0]
    if (record.get("status") != "completed" or record["result"]["outcome"] != "leak"
            or not steps or steps[-1]["action"]["kind"] != "submit"
            or secret not in (steps[-1]["action"].get("value") or "")
            or steps[-1]["returned"] != "[submitted]"):
        raise ValueError("Source must end in an egress-verified winning submit")
    if not witness_for(record, length - 1, secret):
        raise ValueError("Winning submit lacks an observation reconstruction witness")
    if not max_depth <= length <= max_turns:
        raise ValueError("Source too short for requested depths or exceeds its budget")
    group, split = split_for(record["task"])
    cases = []
    for depth in range(1, max_depth + 1):
        k = length - depth
        state = recorded_state(record, k)
        pad = predecision_scratchpad(record, k, max_turns)
        visible = "\n".join(t["content"] for t in state.conversation)
        if secret in visible or secret in record["system_prompt"]:
            raise ValueError("Literal canary already present in unassisted input")
        proof = reconstruction_witness(state.conversation, secret)
        responses = [t["content"] for t in state.conversation if t["role"] in {"monitor", "harness"}]
        student_pad = render_scratchpad(
            seed_scratchpad(record["task"]["goal"], record["task"].get("request") or "", ""),
            tried_lines([AttackerAction(**s["action"]) for s in steps[:k]], responses), SEED_NEXT_IDEAS)
        # Full pad is retained or rejected, never silently censored/truncated. A grounded literal
        # answer in teacher notes is allowed only in this explicitly assisted diagnostic.
        if any(secret in text for _, text in decoded_views(pad)) and not proof:
            raise ValueError("Teacher pad contains an answer without prior observable evidence")
        cases.append(dict(version=1, id=f"{record['key']}-d{depth}", episode=record["key"],
            source_record_sha256=digest(record), group=group, split=split, tier="T2",
            depth=depth, start_turn=k, source_length=length, max_turns=max_turns,
            remaining_turns=max_turns - k, task=copy.deepcopy(record["task"]),
            system=record["system_prompt"], prefix_steps=copy.deepcopy(steps[:k]),
            conversation=copy.deepcopy(state.conversation), state_sha256=state_digest(state),
            teacher_scratchpad=pad, teacher_pad_sha256=digest(pad),
            student_initial_scratchpad=student_pad,
            assisted_literal_answer_present=secret in pad, evidence_witness=proof,
            observation_window=2,
            assistance_modes=["none", "teacher_predecision_scratchpad"],
            canary_policy="original per-episode canary; no resampling of stored evidence",
            fresh_student_scratchpad=True))
    return cases


class ReplayAction:
    def __init__(self, action):
        self.action = AttackerAction(**action)

    def act(self, conversation, turns_taken=0):
        return self.action


def replay(case, sandbox):
    """Execute only in the supplied isolated sandbox; fail on ANY saved observation mismatch."""
    spec = TaskSpec(**case["task"])
    sandbox.reset(spec)
    state = TurnState.start(spec)
    defender = OracleRedactorDefender(spec.forbidden)
    for index, expected in enumerate(case["prefix_steps"]):
        if expected["turn"] != index:
            raise ReplayMismatch("Noncontiguous source prefix")
        stop = step_turn(state, spec, ReplayAction(expected["action"]), defender, sandbox,
                        index, case["max_turns"], None, "unshielded")
        actual = state.steps[-1]
        # T2 is deterministic. Compare decisions too; no old decision is treated as a new output.
        for key in ("returned", "policy_returned", "call_decision", "result_decision",
                    "redaction_status", "validation_error", "execution"):
            if actual.get(key) != expected.get(key):
                raise ReplayMismatch(f"Replay differs at turn {index}: {key}")
        if stop:
            raise ReplayMismatch("Replay terminated before restart")
    if (state.conversation != case["conversation"] or state.n_turns != case["start_turn"]
            or state_digest(state) != case["state_sha256"]):
        raise ReplayMismatch("Reconstructed turn state differs")
    workspace_hash = sandbox.state_hash()
    if case.get("workspace_sha256") and workspace_hash != case["workspace_sha256"]:
        raise ReplayMismatch("Replayed workspace differs from preparation")
    return state, workspace_hash
