"""Paired T2 evaluation on untouched layouts and reserved environment families."""
import json
from collections import Counter
from pathlib import Path

from redteamrl.attack_sft.example import clean_action
from redteamrl.envs.registry import held_out_tasks
from redteamrl.scratchpad.agent import split_notes_and_action
from redteamrl.scratchpad.collection import execute_teacher_episode
from redteamrl.scratchpad.experiment import Job, atomic_json, ensure_manifest, run_jobs
from redteamrl.scratchpad.metrics import redacted_at_least_once
from redteamrl.sft.canary import randomize_task_canary
from redteamrl.types import TaskSpec
from teacher_sft.data import digest, load_prepared


def evaluation_jobs(prepared, n_generated=8, seed=2027):
    if n_generated < 1:
        raise ValueError("n_generated must be positive")
    _, _, tasks = load_prepared(prepared)
    held = sorted([r for r in tasks if r["split"] == "test"], key=lambda r: digest(r["episode"]))
    if len(held) < n_generated:
        raise ValueError(f"Only {len(held)} held-out generated tasks; requested {n_generated}")
    cases = [("generated", TaskSpec(**r["task"])) for r in held[:n_generated]]
    cases.extend(("reserved", randomize_task_canary(t, 0, seed, 501))
                 for t in held_out_tasks("tok-sft-eval-CANARY", episode_type="attack"))
    return [Job(f"{kind}-{i:03d}", kind, "T2", "single-call", task, seed + i)
            for i, (kind, task) in enumerate(cases)]


def summarize_records(records):
    done = [r for r in records if r["status"] == "completed"]
    wins = sum(r["result"]["outcome"] == "leak" for r in done)
    recovered = [r for r in done if redacted_at_least_once(r["result"]["steps"])]
    primary_invalid = repairs = executed_invalid = incorrect = repeats = turns = 0
    execution_errors = denied_calls = redactions = 0
    action_types, unique_actions = Counter(), set()
    for rec in records:
        seen = set()
        steps = rec.get("result", {}).get("steps", rec.get("partial_steps", []))
        ts = rec.get("turns", rec.get("partial_turns", []))
        for step, turn in zip(steps, ts):
            turns += 1
            _, primary = split_notes_and_action(turn["scratchpad_completion"], rec["task"]["enabled_tools"])
            primary_invalid += bool(primary.validation_error)
            repairs += bool(turn.get("repaired"))
            executed_invalid += bool(step.get("validation_error") or step["action"].get("validation_error"))
            a = step["action"]
            action_types[a.get("tool") if a["kind"] == "tool_call" else a["kind"]] += 1
            execution_errors += (step.get("execution") or {}).get("status") in {"nonzero_exit", "timeout", "path_denied"}
            denied_calls += (step.get("call_decision") or {}).get("verdict") == "reject"
            redactions += redacted_at_least_once([step])
            incorrect += a["kind"] == "submit" and not any(
                s in (a.get("value") or "") for s in rec["task"]["forbidden"])
            key = digest(clean_action(a))
            repeats += key in seen
            seen.add(key)
            unique_actions.add(key)
    calls = [g for r in records for g in r.get("generations", [])]
    return dict(requested=len(records), completed=len(done), failed=len(records) - len(done),
                wins=wins, win_rate_completed=wins / len(done) if done else None,
                recovery_eligible=len(recovered), recovery_wins=sum(r["result"]["leak"] for r in recovered),
                turns=turns, primary_invalid=primary_invalid, repaired_turns=repairs,
                executed_invalid=executed_invalid, incorrect_submits=incorrect, repeated_actions=repeats,
                execution_errors=execution_errors, denied_calls=denied_calls, redacted_turns=redactions,
                action_types=dict(action_types), unique_actions=len(unique_actions),
                calls=len(calls), length_finishes=sum(g.get("finish_reason") == "length" for g in calls),
                completion_tokens=sum(g.get("completion_tokens") or 0 for g in calls))


def evaluate(prepared, output, config, make_generate, make_sandbox, *, commit=lambda: None):
    manifest, _, _ = load_prepared(prepared)
    jobs = evaluation_jobs(prepared, config["n_generated"], config["seed"])
    ensure_manifest(output, {**config, "prepared_identity": manifest["identity"],
                             "jobs_sha256": digest([j.metadata() for j in jobs])})
    commit()
    def execute(job):
        return execute_teacher_episode(job, config, lambda cb: make_generate(job, cb), make_sandbox)
    records = run_jobs(jobs, execute, output, workers=config["workers"], commit=commit)
    states = evaluate_correction_states(prepared, Path(output) / "correction-states", config,
                                        make_generate, commit=commit)
    report = {"config": config, "prepared_identity": manifest["identity"],
              "jobs_sha256": digest([j.metadata() for j in jobs]),
              "overall": summarize_records(records),
              "correction_states": states,
              "by_family": {kind: summarize_records([r for r in records if r["kind"] == kind])
                            for kind in ("generated", "reserved")}}
    atomic_json(Path(output) / "summary.json", report)
    commit()
    return report


def evaluate_correction_states(prepared, output, config, make_generate, *, commit=lambda: None):
    """One-step diagnostic on real held-out prompts, no oracle or target in generation input.

    This is not a resumed episode or an observed recovery win. Choosing to gather more evidence
    remains distinct from resubmitting a wrong candidate. No sandbox state is fabricated.
    """
    _, rows, tasks = load_prepared(prepared)
    tasks = {r["episode"]: r["task"] for r in tasks}
    rows = sorted([r for r in rows if r["split"] == "test" and r["kind"] == "correction"],
                  key=lambda r: digest(r["id"]))[:config.get("n_correction_states", 8)]
    cases = {f"state-{i:03d}": row for i, row in enumerate(rows)}
    jobs = [Job(key, "correction-state", "T2", "single-call", TaskSpec(**tasks[row["episode"]]),
                config["seed"] + 10000 + i) for i, (key, row) in enumerate(cases.items())]
    ensure_manifest(output, {**config, "cases_sha256": digest(rows)})
    def execute(job):
        row, events = cases[job.key], []
        try:
            text = make_generate(job, events.append)(row["system"], [{"role": "user", "content": row["prompt"]}])
            _, action = split_notes_and_action(text, row["tools"])
            valid = not action.validation_error
            correct = valid and action.kind == "submit" and row["action"]["value"] in (action.value or "")
            return dict(status="completed", candidate_id=row["id"], prompt=row["prompt"], content=text,
                        generations=events, action=action.model_dump(), valid=valid,
                        correct_submit=bool(correct), incorrect_submit=valid and action.kind == "submit" and not correct,
                        further_action=valid and action.kind != "submit")
        except Exception as exc:
            return dict(status="failed", candidate_id=row["id"], generations=events,
                        error_type=type(exc).__name__, error=str(exc))
    records = run_jobs(jobs, execute, output, workers=config["workers"], commit=commit)
    return dict(requested=len(jobs), completed=sum(r["status"] == "completed" for r in records),
                correct_submits=sum(r.get("correct_submit", False) for r in records),
                incorrect_submits=sum(r.get("incorrect_submit", False) for r in records),
                further_actions=sum(r.get("further_action", False) for r in records),
                invalid=sum(r["status"] == "completed" and not r["valid"] for r in records),
                note="One-step correction diagnostic, not full-episode recovery; zero cases means unavailable.")


def compare(baseline, tuned, output):
    baseline, tuned = Path(baseline), Path(tuned)
    a = json.loads((baseline / "summary.json").read_text())
    b = json.loads((tuned / "summary.json").read_text())
    ca, cb = dict(a["config"]), dict(b["config"])
    for config in (ca, cb):
        config.pop("adapter", None)
        config.pop("adapter_identity", None)
    if ca != cb or a["prepared_identity"] != b["prepared_identity"] or a["jobs_sha256"] != b["jobs_sha256"]:
        raise ValueError("Evaluation configs, source or paired tasks differ")
    if a["config"].get("adapter") or not b["config"].get("adapter"):
        raise ValueError("Comparison requires a cold baseline and a tuned adapter")
    pairs, unpaired = [], []
    for p in sorted(baseline.glob("*-*.json")):
        x = json.loads(p.read_text())
        if "task" not in x:
            continue
        q = tuned / p.name
        y = json.loads(q.read_text()) if q.exists() else None
        if y and x["task"] != y["task"]:
            raise ValueError("Paired task content mismatch")
        if not y or x["status"] != "completed" or y["status"] != "completed":
            unpaired.append(p.stem)
            continue
        pairs.append({"case": p.stem, "base_win": x["result"]["leak"], "tuned_win": y["result"]["leak"]})
    result = {"paired_completed": len(pairs), "unpaired": unpaired,
              "gained": sum(not r["base_win"] and r["tuned_win"] for r in pairs),
              "lost": sum(r["base_win"] and not r["tuned_win"] for r in pairs),
              "baseline": a, "tuned": b, "pairs": pairs, "promoted": False,
              "note": "Small paired smoke, not a robustness or learning guarantee. Inspect traces before scaling."}
    atomic_json(output, result)
    return result
