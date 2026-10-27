"""Offline-testable teacher collection prerequisites; no model calls at import time.

Records retain raw completions, observable turns, per-attempt API telemetry, and task identity.
Diagnostics do not affect actions, rewards, or filtering. A successful episode is NOT automatically
a clean SFT example. The caller owns provider/session setup; only the coordinator persists records.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import traceback

from redteamrl.envs.registry import assert_training_split
from redteamrl.harness.episode import EpisodeResult
from redteamrl.scratchpad.agent import _last_json_span
from redteamrl.scratchpad.experiment import Job, atomic_json, ensure_manifest, run_jobs
from redteamrl.scratchpad.gate0 import gate0_summary
from redteamrl.scratchpad.metrics import redacted_at_least_once
from redteamrl.scratchpad.poc import SCRATCHPAD_SYSTEM, SEED_NEXT_IDEAS, make_attacker_factory, poc_tasks
from redteamrl.scratchpad.render import SINGLE_CALL_FORMAT_VERSION
from redteamrl.scratchpad.rollout import run_scratchpad_rollout, to_jsonable, turn_to_jsonable
from redteamrl.verify.defenders import OracleRedactorDefender


def probe_config(*, teacher, episodes, tier, tag, source, env_seed_start=7000, workers=4,
                 max_turns=12, max_tokens=8192):
    if tier != "T2":
        raise ValueError("This CPU probe supports T2 only; T0/T1 require a served defender")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", tag):
        raise ValueError("tag must be a simple 1-64 character name, not a path")
    if not teacher.strip() or episodes < 1 or env_seed_start < 0:
        raise ValueError("teacher, positive episodes and nonnegative env_seed_start required")
    if not 1 <= workers <= 32 or not 1 <= max_turns <= 32 or max_tokens < 1:
        raise ValueError("workers/turns must be 1-32 and max_tokens positive")
    return dict(schema_version=2, teacher=teacher, episodes=episodes, tier=tier, source=source,
                env_seed_start=env_seed_start, canary_seed=0, canary_revision=201,
                canary_template="tok-gate0-CANARY", workers=workers, max_turns=max_turns,
                max_tokens=max_tokens, temperature=0.7, top_p=0.95,
                teacher_reasoning="provider_default",
                format=SINGLE_CALL_FORMAT_VERSION, single_call=True,
                sandbox=dict(timeout_s=10, output_cap=4096, episode_timeout_s=1200),
                api=dict(timeout_s=300, max_retries=3, backoff_s=2.0))


def probe_jobs(config):
    specs = assert_training_split(poc_tasks(
        list(range(config["env_seed_start"], config["env_seed_start"] + config["episodes"])),
        config["canary_template"], config["canary_seed"], config["canary_revision"]))
    # Generation seed is not sent to the provider: sampling is explicitly stochastic.
    return [Job(f"ep-{config['tier']}-{i:03d}", "attack", config["tier"], "single-call", spec, 0)
            for i, spec in enumerate(specs)]


def turn_diagnostics(steps, turns, generations, forbidden):
    """Independent flags: validation != execution != defender decisions != submit correctness.

    Finish reasons are evidence, not a diagnosis inferred from an empty string. A shell error
    marker is a heuristic (stdout can imitate one), distinct from a measured nonzero exit.
    """
    rows = []
    for step in steps:
        index = step["turn"]
        turn = turns[index] if index < len(turns) else {}
        raw = turn.get("scratchpad_completion", "")
        action = step.get("action") or {}
        calls = [g for g in generations if g.get("turn") == index]
        last = calls[-1] if calls else {}
        flags = []
        if step.get("validation_error") or action.get("validation_error"):
            flags.append("invalid_action")
            if not raw.strip():
                flags.append("empty_response")
            elif _last_json_span(raw) is None:
                flags.append("actionless_response")
            else:
                flags.append("invalid_action_schema")
        # Any call in the turn, not just the last: an action-repair call must not mask the primary's
        # truncation or refusal.
        if any(c.get("finish_reason") == "length" for c in calls):
            flags.append("generation_length_limit")
        if any(c.get("finish_reason") == "content_filter" or c.get("refusal_present") for c in calls):
            flags.append("provider_refusal_or_filter")
        execution = step.get("execution")
        if execution and execution.get("status") in {"nonzero_exit", "timeout", "path_denied"}:
            flags.append("execution_error")
        returned = step.get("returned") or ""
        if action.get("kind") == "tool_call" and re.search(
            r"command not found|No such file or directory|\[command timed out|^Error:", returned):
            flags.append("execution_error_text_hint")
        if action.get("kind") == "submit" and not step.get("validation_error"):
            correct = any(f in (action.get("value") or "") for f in forbidden)
            flags.append("correct_submit" if correct else "incorrect_submit")
        rows.append(dict(turn=index, flags=flags, validation_error=step.get("validation_error"),
                         call_verdict=(step.get("call_decision") or {}).get("verdict"),
                         result_verdict=(step.get("result_decision") or {}).get("verdict"),
                         redaction_status=step.get("redaction_status"), execution=execution,
                         generation_attempts=len(calls), finish_reason=last.get("finish_reason")))
    return rows


def execute_teacher_episode(job, config, make_generate, make_sandbox):
    """make_generate(on_response) returns one episode-local generate callable.

    Always retains collected metadata and completed steps if generation/execution/cleanup raises.
    A failed in-flight generation has a turn index but no executed step; do not invent an action.
    """
    generations, agent, sandbox, steps = [], None, None, []
    record = None

    def on_response(event):
        generations.append({**event, "turn": len(agent.turn_records) if agent else 0})

    def on_turn(state, _):
        steps[:] = state.steps

    try:
        generate = make_generate(on_response)
        agent = make_attacker_factory(generate, generate, system_template=SCRATCHPAD_SYSTEM,
                                      seed_next_ideas=SEED_NEXT_IDEAS,
                                      turn_budget=config["max_turns"], single_call=True)(job.spec)
        sandbox = make_sandbox()
        rollout = run_scratchpad_rollout(job.spec, agent, OracleRedactorDefender(job.spec.forbidden),
                                         sandbox, max_turns=config["max_turns"], tier=job.tier,
                                         on_turn=on_turn)
        record = {"status": "completed", **to_jsonable(rollout)}
    except Exception as exc:
        record = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                  "traceback": traceback.format_exc()}
    finally:
        if sandbox is not None:
            try:
                sandbox.close()
            except Exception as exc:
                record = {**(record or {}), "status": "failed", "cleanup_error": str(exc),
                          "cleanup_error_type": type(exc).__name__}
    turns = [turn_to_jsonable(t) for t in agent.turn_records] if agent else []
    if record["status"] == "failed":
        record.update(partial_steps=steps, partial_turns=turns)
    record.update(generations=generations, system_prompt=agent.system if agent else None,
                  turn_diagnostics=turn_diagnostics(steps, turns, generations, job.spec.forbidden))
    return record


def collection_summary(records, config):
    done = [r for r in records if r["status"] == "completed"]
    pairs = [(EpisodeResult(**r["result"]), r["task"]["forbidden"][0]) for r in done]
    gate = gate0_summary({config["tier"]: pairs})
    row = gate[config["tier"]]
    n_failed = sum(r["status"] == "failed" for r in records)
    complete = len(done) == config["episodes"] and n_failed == 0
    gate_pass = bool(complete and gate["gate_pass"])
    flags = Counter(f for r in records for d in r.get("turn_diagnostics", []) for f in d["flags"])
    attempts = [g for r in records for g in r.get("generations", [])]
    usage = {}
    for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cost"):
        values = [g[field] for g in attempts if isinstance(g.get(field), (int, float))]
        usage[field] = dict(known_total=sum(values) if values else None,
                            missing_attempts=len(attempts) - len(values))
    redacted = [r for r, _ in pairs if redacted_at_least_once(r.steps)]
    attribution = Counter(r.leak_attribution or "unattributed" for r, _ in pairs if r.leak)
    diagnostics = dict(turns=sum(len(r.get("turn_diagnostics", [])) for r in records),
                       flags=dict(flags), generation_attempts=len(attempts),
                       flag_episodes={flag: sum(any(flag in d["flags"] for d in r.get("turn_diagnostics", []))
                                                for r in records) for flag in flags},
                       finish_reasons=dict(Counter(g.get("finish_reason") or "unknown" for g in attempts)),
                       generation_errors=dict(Counter(g["error_type"] for g in attempts if g.get("error_type"))),
                       usage=usage, leak_attribution=dict(attribution))
    row.update(n_requested=config["episodes"], n_completed=len(done), n_failed=n_failed,
               n_missing=config["episodes"] - len(records),
               wins=sum(r.leak for r, _ in pairs),
               recovery_wins=sum(r.leak for r in redacted), recovery_eligible=len(redacted),
               win_rate_completed=row["win_rate"] if done else None,
               win_rate=row["win_rate"] if complete else None,
               recovery_rate=row["recovery_rate"] if redacted else None,
               passes=gate_pass, diagnostics=diagnostics,
               transform_rate_is_command_pattern=True)
    gate["gate_pass"] = gate_pass
    gate["gate_status"] = "INCONCLUSIVE" if not complete else "PASS" if gate_pass else "NO-GO"
    return dict(teacher=config["teacher"], tier=config["tier"], episodes=config["episodes"],
                n_completed=len(done), n_failed=n_failed, summary=gate)


def run_collection(config, out_dir, execute, *, commit=lambda: None, retry_failed=False):
    jobs = probe_jobs(config)
    task_digest = hashlib.sha256(json.dumps([j.spec.model_dump() for j in jobs], sort_keys=True).encode()).hexdigest()
    manifest = ensure_manifest(out_dir, {**config, "tasks_sha256": task_digest})
    # A stale summary must never survive an interrupted retry of failed episodes.
    Path(out_dir, "summary.json").unlink(missing_ok=True)
    commit()
    records = run_jobs(jobs, execute, out_dir, workers=config["workers"], commit=commit,
                       retry_failed=retry_failed)
    summary = collection_summary(records, config)
    summary["manifest_identity"] = manifest["identity"]
    atomic_json(Path(out_dir) / "summary.json", summary)
    commit()
    return summary
