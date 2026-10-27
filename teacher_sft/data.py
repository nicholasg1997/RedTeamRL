"""Immutable harvest import, split assignment, evidence witnesses and candidate selection.

No model calls; no recorded shell commands are executed. Hindsight corrections are labelled
counterfactual submit targets, never credited as observed teacher wins. Prose screening is a
conservative heuristic, not a proof of semantic grounding.
"""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from redteamrl.attack_sft.example import clean_action
from redteamrl.envs.registry import assert_training_split
from redteamrl.harness.protocol import validate_action
from redteamrl.scratchpad.agent import _last_json_span, split_notes_and_action
from redteamrl.scratchpad.experiment import atomic_json
from redteamrl.types import TaskSpec
from teacher_sft import VERSION


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    temporary.replace(path)


def split_for(task):
    # Same filesystem layout stays together even across seeds, canaries and derived examples.
    group = digest(sorted(f["path"] for f in task["seed_files"]))
    bucket = int(group[:8], 16) % 100
    return group, "train" if bucket < 80 else "dev" if bucket < 90 else "test"


def decoded_views(text):
    """Bounded exact decoders over observed text, independent of the oracle key."""
    yield "literal", text
    tokens = re.findall(r"[A-Za-z0-9+/]{12,}={0,2}", text)
    candidates = set(tokens)
    # Standard base64 wraps on complete lines. Avoid arbitrary substring search using the key.
    block = []
    for line in [*text.splitlines(), ""]:
        if re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", line.strip()):
            block.append(line.strip())
        else:
            if block:
                candidates.add("".join(block))
                block = []
    for token in candidates:
        try:
            yield "base64", base64.b64decode(token, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            pass
    chars, hexes = [], []
    for line in text.splitlines():
        parts = re.sub(r"^\s*[0-7]{7}\s+", "", line).rsplit(":", 1)[-1].split()
        if len(parts) >= 2 and all(len(p) == 1 or p in {r"\n", r"\t", r"\r", r"\0"} for p in parts):
            chars.extend(p if len(p) == 1 else "\n" for p in parts)
        else:
            chars.append("\n")
        if parts and all(re.fullmatch(r"[0-9a-fA-F]{2}", p) for p in parts):
            hexes.extend(parts)
        else:
            hexes.append("00")
    yield "character_columns", "".join(chars)
    yield "hex", bytes.fromhex("".join(hexes)).decode("utf-8", errors="replace")


def visible_observations(record, index):
    """Only actual observations present in full in the exact saved pre-action prompt.

    A model-authored claim in the scratchpad alone is never a reconstruction witness.
    """
    prompt = record["turns"][index]["scratchpad_prompt"]
    return [(s["turn"], s["returned"]) for s in record["result"]["steps"][:index]
            if s.get("returned") and s["returned"] in prompt]


def witness_for(record, index, secret):
    for turn, text in visible_observations(record, index):
        for method, decoded in decoded_views(text):
            if secret in decoded:
                return {"observation_turn": turn, "method": method, "observation_sha256": digest(text)}
    return None


def import_harvest(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    expected = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if manifest.get("identity") != expected:
        raise ValueError("Harvest manifest identity mismatch")
    if config.get("tier") != "T2" or not config.get("single_call"):
        raise ValueError("v1 importer requires the audited single-call T2 collector")
    from redteamrl.scratchpad.collection import probe_jobs
    jobs = probe_jobs(config)
    tasks_sha = hashlib.sha256(json.dumps([j.spec.model_dump() for j in jobs], sort_keys=True).encode()).hexdigest()
    if tasks_sha != config.get("tasks_sha256"):
        raise ValueError("Task generator differs from the original manifest; use its source snapshot")
    records, rejected = [], []
    for job in jobs:
        path = root / (job.key + ".json")
        if not path.exists():
            rejected.append({"episode": job.key, "reason": "missing_record"})
            continue
        rec = json.loads(path.read_text())
        if rec.get("key") != job.key or rec.get("task") != job.spec.model_dump():
            raise ValueError(f"Episode identity mismatch: {path}")
        if rec.get("status") != "completed":
            rejected.append({"episode": job.key, "reason": "exception_failed"})
            continue
        assert_training_split([TaskSpec(**rec["task"])])
        steps, turns = rec["result"]["steps"], rec["turns"]
        if len(steps) != len(turns) or len(steps) != rec["result"]["n_turns"]:
            raise ValueError(f"Incomplete turn alignment: {path}")
        matched = False
        for i, (s, t) in enumerate(zip(steps, turns)):
            if s["turn"] != i or clean_action(s["action"]) != clean_action(t["action"]):
                raise ValueError(f"Action/turn mismatch: {path}:{i}")
            matched |= any(f in (s.get("returned") or "") for f in job.spec.forbidden)
            if s["action"]["kind"] == "submit":
                correct = any(f in (s["action"].get("value") or "") for f in job.spec.forbidden)
                if (s["returned"] == "[submitted]") != correct:
                    raise ValueError(f"Submit mismatch: {path}:{i}")
                matched |= correct
        if matched != rec["result"]["leak"] or matched != (rec["result"]["outcome"] == "leak"):
            raise ValueError(f"Outcome mismatch: {path}")
        if not rec.get("system_prompt"):
            raise ValueError(f"Missing saved system prompt: {path}")
        records.append(rec)
    return manifest, records, rejected


def select_candidates(records):
    rows, dropped = [], []
    for rec in records:
        group, split = split_for(rec["task"])
        secret = rec["task"]["forbidden"][0]
        seen = set()
        correction_added = False
        for i, (step, turn) in enumerate(zip(rec["result"]["steps"], rec["turns"])):
            common = dict(episode=rec["key"], turn=i, group=group, split=split,
                          task_id=rec["task"]["id"], system=rec["system_prompt"],
                          prompt=turn["scratchpad_prompt"], tools=rec["task"]["enabled_tools"],
                          original_outcome=rec["result"]["outcome"], record_sha256=digest(rec))
            action = clean_action(step["action"])
            proof = witness_for(rec, i, secret)
            prior_wrong = any(s["action"]["kind"] == "submit" and
                              secret not in (s["action"].get("value") or "")
                              for s in rec["result"]["steps"][:i])
            # Only a state AFTER observed rejection, with evidence still visible. This labels a
            # counterfactual submit; it never edits the trajectory or credits a teacher win.
            if rec["result"]["outcome"] != "leak" and prior_wrong and proof and not correction_added:
                corrected = {"kind": "submit", "value": secret}
                rows.append({**common, "id": f"{rec['key']}:{i}:correction", "kind": "correction",
                             "action": corrected, "witness": proof, "teacher_notes": None,
                             "target_source": "evidence_checked_counterfactual_submit"})
                correction_added = True
                continue
            reason = None
            notes, parsed = split_notes_and_action(turn["scratchpad_completion"], common["tools"])
            if rec["result"]["outcome"] != "leak":
                reason = "loss_context_only"
            elif step.get("validation_error") or parsed.validation_error:
                reason = "invalid_primary_or_repaired_turn"
            elif turn.get("repaired") or turn.get("action_prompt"):
                reason = "repair_requires_separate_authoring"
            elif not notes.strip().startswith("## "):
                reason = "missing_notes"
            elif (step.get("execution") or {}).get("status") in {"nonzero_exit", "timeout", "path_denied"}:
                reason = "execution_error"
            elif action["kind"] == "tool_call" and (step.get("call_decision") or {}).get("verdict") != "allow":
                reason = "rejected_tool"
            elif action["kind"] == "message":
                reason = "message_not_a_validated_progress_target"
            elif action["kind"] == "submit" and (secret not in action.get("value", "") or not proof):
                reason = "submit_without_visible_evidence"
            elif clean_action(parsed.model_dump()) != action:
                reason = "primary_action_differs"
            key = digest(action)
            if key in seen:
                reason = "repeated_action"
            seen.add(key)
            if reason:
                dropped.append({"episode": rec["key"], "turn": i, "reason": reason})
                continue
            rows.append({**common, "id": f"{rec['key']}:{i}:natural", "kind": "natural",
                         "action": action, "witness": proof, "teacher_notes": notes,
                         "target_source": "observed_winning_episode_action"})
    return rows, dropped


def prepare(root, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepared dataset directory must be new; never overwrite split assignments")
    manifest, records, rejected = import_harvest(root)
    rows, dropped = select_candidates(records)
    tasks = [{"episode": r["key"], "task": r["task"], "group": split_for(r["task"])[0],
              "split": split_for(r["task"])[1]} for r in records]
    identity = {"version": VERSION, "harvest_identity": manifest["identity"],
                "records": {r["key"]: digest(r) for r in records},
                "candidate_digest": digest(rows), "tasks_digest": digest(tasks),
                "split_rule": "sha256(sorted-seed-file-paths)-80/10/10-v1"}
    atomic_json(output / "manifest.json", {"identity": digest(identity), "config": identity})
    write_jsonl(output / "candidates.jsonl", rows)
    write_jsonl(output / "tasks.jsonl", tasks)
    write_jsonl(output / "rejections.jsonl", rejected + dropped)
    summary = {"episodes_completed": len(records), "episodes_omitted": len(rejected),
               "candidates": len(rows), "by_split_kind": dict(Counter(f"{r['split']}/{r['kind']}" for r in rows)),
               "rejections": dict(Counter(r["reason"] for r in rejected + dropped)),
               "notes": "Candidates need student authoring and screening; this is not a train-ready dataset."}
    atomic_json(output / "summary.json", summary)
    return summary


def load_prepared(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rows, tasks = read_jsonl(root / "candidates.jsonl"), read_jsonl(root / "tasks.jsonl")
    if digest(manifest["config"]) != manifest["identity"] or digest(rows) != manifest["config"]["candidate_digest"] \
            or digest(tasks) != manifest["config"]["tasks_digest"]:
        raise ValueError("Prepared data integrity check failed")
    return manifest, rows, tasks
