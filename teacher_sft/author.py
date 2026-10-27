"""Student-owned, target-conditioned reasoning; never a claim of unaided student success."""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path

from redteamrl.attack_sft.example import clean_action
from redteamrl.scratchpad.agent import _last_json_span, split_notes_and_action
from redteamrl.scratchpad.experiment import atomic_json, ensure_manifest
from teacher_sft.data import decoded_views, digest, load_prepared, read_jsonl, write_jsonl

INSTRUCTION = """[TRAINING AUTHORING ONLY — this suffix is removed from the training input]
The selected next action is supplied below. Reason from the original observations to this action.
Do not claim a future outcome, privileged answer, or that an unverified memory entry is an observation.
If a prior candidate conflicts with evidence, distinguish the two and revise the unsupported claim.
If this action cannot be justified, respond with UNSUPPORTED instead of inventing a justification.
Write updated scratchpad notes as ## sections, then the exact selected JSON action on the final line.
The model's private reasoning is generated normally; no teacher reasoning is supplied.
Selected action:
"""


def screen_text(text, prompt):
    """Reject obvious hindsight/format leaks and unsupported high-entropy values.

    Passing this lexical screen does not certify arbitrary factual prose. Exported review samples
    and independent behavioral evaluation are necessary to assess that residual risk.
    """
    banned = ("ground truth", "oracle answer", "in hindsight", "eventually succeeded",
              "training authoring", "selected action is supplied", "future outcome")
    if any(s in text.lower() for s in banned):
        return "privileged_authoring_reference"
    evidence = prompt + "\n" + "\n".join(v for _, v in decoded_views(prompt))
    for value in re.findall(r"\b[A-Za-z0-9_-]{20,}\b", text):
        if value not in evidence and any(c.isdigit() for c in value):
            return "unsupported_value"
    return None


def validate_authored(candidate, reasoning, content):
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None, "missing_student_reasoning"
    if not isinstance(content, str) or not content.strip():
        return None, "missing_final_content"
    if any(t in reasoning + content for t in ("<think>", "</think>", "<|im_end|>", "<|im_start|>")):
        return None, "embedded_control_token"
    span = _last_json_span(content)
    notes, action = split_notes_and_action(content, candidate["tools"])
    if not span or content[span[1]:].strip() or action.validation_error or not notes.startswith("## "):
        return None, "bad_notes_action_format"
    if clean_action(action.model_dump()) != candidate["action"]:
        return None, "changed_selected_action"
    reason = screen_text(reasoning + "\n" + notes, candidate["prompt"])
    if reason:
        return None, reason
    final = notes + "\n" + json.dumps(candidate["action"], ensure_ascii=False)
    return {"id": candidate["id"], "episode": candidate["episode"], "group": candidate["group"],
            "split": candidate["split"], "kind": candidate["kind"], "system": candidate["system"],
            "prompt": candidate["prompt"], "reasoning": reasoning.strip(), "final": final,
            "action": candidate["action"], "tools": candidate["tools"],
            "witness": candidate["witness"], "candidate_sha256": digest(candidate),
            "grounding": "mechanical_screen_not_semantic_proof",
            "reasoning_source": "student_target_conditioned"}, None


def author_dataset(prepared, output, generate, config, *, commit=lambda: None):
    """generate(system, prompt) -> (reasoning, final_content, metadata). No test authoring.

    Each candidate is checkpointed separately; failed authoring is retained, not silently retried.
    Use a fresh output tag for a new sampling attempt, preserving unsuccessful authoring records.
    """
    manifest, candidates, _ = load_prepared(prepared)
    output = Path(output)
    ensure_manifest(output, {**config, "prepared_identity": manifest["identity"]})
    commit()
    accepted, attempts, pending = [], {}, []
    limit = config.get("max_candidates", 128)
    if limit < 1:
        raise ValueError("max_candidates must be positive")
    eligible = [r for r in candidates if r["split"] != "test"]
    # A bounded smoke samples both kinds/splits rather than taking the first episode's turns.
    eligible.sort(key=lambda r: digest(r["id"]))
    eligible = eligible[:limit]
    def author_one(row):
        prompt = row["prompt"] + "\n\n" + INSTRUCTION + json.dumps(row["action"])
        attempt = {"id": row["id"], "candidate_sha256": digest(row), "authoring_prompt": prompt}
        try:
            reasoning, content, meta = generate(row["system"], prompt)
            example, rejection = validate_authored(row, reasoning, content)
            if meta.get("finish_reason") != "stop":
                example, rejection = None, "incomplete_generation"
            attempt.update(reasoning=reasoning, content=content, generation=meta,
                           example=example, rejection=rejection)
        except Exception as exc:
            attempt.update(example=None, rejection="generation_exception", error=str(exc),
                           error_type=type(exc).__name__)
        return attempt

    for row in eligible:
        path = output / (digest(row["id"]) + ".json")
        if path.exists():
            attempt = json.loads(path.read_text())
            if attempt["candidate_sha256"] != digest(row):
                raise ValueError("Authoring resume candidate mismatch")
            attempts[row["id"]] = attempt
        else:
            pending.append(row)
    workers = config.get("workers", 4)
    if not 1 <= workers <= 16:
        raise ValueError("Author workers must be 1-16")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(author_one, row): row for row in pending}
        for future in as_completed(futures):
            row, attempt = futures[future], future.result()
            attempts[row["id"]] = attempt
            atomic_json(output / (digest(row["id"]) + ".json"), attempt)
            commit()  # Coordinator only: workers never commit the volume.
            print(f"author {len(attempts)}/{len(eligible)} {row['id']}: "
                  f"{attempt['rejection'] or 'accepted'}", flush=True)
    for row in eligible:
        attempt = attempts[row["id"]]
        if attempt["example"]:
            accepted.append(attempt["example"])
    write_jsonl(output / "examples.jsonl", accepted)
    artifact = {"config": config, "prepared_identity": manifest["identity"],
                "examples_sha256": digest(accepted), "attempted": len(attempts), "accepted": len(accepted),
                "by_split_kind": dict(Counter(f"{r['split']}/{r['kind']}" for r in accepted)),
                "rejections": dict(Counter(r["rejection"] for r in attempts.values() if r["rejection"]))}
    atomic_json(output / "dataset.json", artifact)
    write_jsonl(output / "review.jsonl", accepted[:20])
    commit()
    return artifact


def load_examples(root):
    root = Path(root)
    meta = json.loads((root / "dataset.json").read_text())
    ensure_manifest(root, {**meta["config"], "prepared_identity": meta["prepared_identity"]})
    examples = read_jsonl(root / "examples.jsonl")
    if digest(examples) != meta["examples_sha256"]:
        raise ValueError("Authored examples digest mismatch")
    seen, ids = {}, set()
    for row in examples:
        if row["split"] not in {"train", "dev"}:
            raise ValueError("Only train/dev examples may enter authoring/training")
        if row["group"] in seen and seen[row["group"]] != row["split"]:
            raise ValueError("Layout leakage across train/dev split")
        if row["kind"] not in {"natural", "correction"} or row["id"] in ids:
            raise ValueError("Unknown example kind or duplicate example id")
        ids.add(row["id"])
        seen[row["group"]] = row["split"]
    return meta, examples
