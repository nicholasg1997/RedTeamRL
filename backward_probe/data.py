"""Freeze training-world source selection and preflight every restart on CPU."""
import json
from pathlib import Path

from redteamrl.scratchpad.experiment import atomic_json, ensure_manifest
from teacher_sft.data import digest, import_harvest, read_jsonl, split_for, write_jsonl
from backward_probe.state import make_cases, replay


def select_sources(records, max_turns, demos=4, max_depth=3):
    if not 1 <= demos <= 16 or not 1 <= max_depth <= 12:
        raise ValueError("demos must be 1-16 and max_depth 1-12")
    selected, rejected, groups, secrets = [], [], set(), set()
    for record in sorted(records, key=lambda r: digest(r["key"])):
        group, split = split_for(record["task"])
        if split != "train":
            rejected.append(dict(episode=record["key"], reason="reserved_split", split=split))
            continue
        try:
            cases = make_cases(record, max_turns, max_depth)
        except ValueError as exc:
            rejected.append(dict(episode=record["key"], reason=str(exc)))
            continue
        secret = record["task"]["forbidden"][0]
        if group in groups or secret in secrets:
            rejected.append(dict(episode=record["key"], reason="duplicate_layout_or_canary"))
            continue
        if len(selected) < demos:
            selected.append((record, cases))
            groups.add(group)
            secrets.add(secret)
    if len(selected) < demos:
        raise ValueError(f"Only {len(selected)} eligible distinct training demos; requested {demos}. "
                         "Reduce --demos or inspect the harvest; never borrow evaluation cases.")
    return selected, rejected


def prepare(harvest, output, config, make_sandbox, *, commit=lambda: None):
    root = Path(output)
    source, records, omissions = import_harvest(harvest)
    selected, rejected = select_sources(records, source["config"]["max_turns"],
                                       config["demos"], config["max_depth"])
    cases = [c for _, rows in selected for c in rows]
    manifest = ensure_manifest(root, {**config, "harvest_identity": source["identity"],
        "sources_sha256": digest([r for r, _ in selected]), "cases_sha256": digest(cases),
        "harvest_sandbox": source["config"]["sandbox"]})
    write_jsonl(root / "selection.jsonl", cases)
    atomic_json(root / "exclusions.json", omissions + rejected)
    commit()
    valid, rejected_replays = [], []
    for case in cases:
        path = root / "preflight" / (case["id"] + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            if result["case_sha256"] != digest(case):
                raise ValueError("Preflight resume identity differs")
        else:
            sandbox, result = None, None
            try:
                sandbox = make_sandbox(source["config"]["sandbox"])
                _, workspace_hash = replay(case, sandbox)
                result = dict(status="ready", workspace_sha256=workspace_hash)
            except Exception as exc:
                result = dict(status="rejected", error_type=type(exc).__name__, error=str(exc))
            finally:
                if sandbox is not None:
                    try:
                        sandbox.close()
                    except Exception as exc:
                        result = dict(status="rejected", error_type="CleanupError", error=str(exc))
            result.update(case_sha256=digest(case), case_id=case["id"])
            atomic_json(path, result)
            commit()
            print(f"preflight {case['id']}: {result['status']} {result.get('error', '')}", flush=True)
        if result["status"] == "ready":
            valid.append({**case, "workspace_sha256": result["workspace_sha256"]})
        else:
            rejected_replays.append(result)
    write_jsonl(root / "cases.jsonl", valid)
    summary = dict(purpose="Inference-only restart feasibility; no learning/fresh-episode claim",
        selection_identity=manifest["identity"], cases_sha256=digest(valid),
        selected_demos=len(selected), requested_states=len(cases), ready_states=len(valid),
        states_by_depth={str(d): dict(selected=sum(c["depth"] == d for c in cases),
                                     ready=sum(c["depth"] == d for c in valid))
                         for d in range(1, config["max_depth"] + 1)},
        rejected_replays=rejected_replays, arms=["unassisted", "teacher-pad"],
        samples=config["samples"], requested_continuations=len(cases) * 2 * config["samples"],
        runnable_continuations=len(valid) * 2 * config["samples"],
        max_student_turns=sum(c["remaining_turns"] for c in valid) * 2 * config["samples"],
        max_calls_per_turn=2, model=config["model"], revision=config["revision"],
        ready=bool(valid), training=False, live_teacher_calls=0)
    atomic_json(root / "preparation.json", summary)
    commit()
    return summary


def load_prepared(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    ensure_manifest(root, manifest["config"])
    report = json.loads((root / "preparation.json").read_text())
    cases = read_jsonl(root / "cases.jsonl")
    if report["selection_identity"] != manifest["identity"] or digest(cases) != report["cases_sha256"]:
        raise ValueError("Prepared restart data changed")
    if any(c["split"] != "train" for c in cases):
        raise ValueError("Only training-world sources are permitted")
    selected = read_jsonl(root / "selection.jsonl")
    if digest(selected) != manifest["config"]["cases_sha256"]:
        raise ValueError("Source selection changed")
    original = {c["id"]: c for c in selected}
    for case in cases:
        if {k: v for k, v in case.items() if k != "workspace_sha256"} != original.get(case["id"]):
            raise ValueError("Prepared case differs from immutable source selection")
    if not cases:
        raise ValueError("No replay-verified restarts; inspect CPU preparation before GPU allocation")
    return manifest, cases, report
