"""Select real train-only evidence/submit pairs without new authoring or shell execution."""
from collections import Counter
import json
from pathlib import Path
import re

from redteamrl.scratchpad.experiment import atomic_json, ensure_manifest
from teacher_sft.author import load_examples
from teacher_sft.data import digest, load_prepared, read_jsonl, write_jsonl
from teacher_sft.encoding import encode_example

AUTHORING_REFERENCE = re.compile(
    r"(?:selected|supplied|provided|given) (?:next |JSON |exact )?action|"
    r"action (?:supplied|provided|given)|training authoring|in hindsight|ground truth", re.I)


def choose_cases(examples, candidates, tasks, tokenizer, *, pairs=4, max_length=8192, max_new_tokens=3072):
    if not 2 <= pairs <= 8:
        raise ValueError("pairs must be 2-8; default four pairs gives eight decisions")
    originals = {r["id"]: r for r in candidates}
    tasks = {r["episode"]: r for r in tasks}
    eligible, rejected = {}, Counter()
    for row in examples:
        if row["split"] != "train" or row["kind"] != "natural":
            continue
        original = originals.get(row["id"])
        if not original or digest(original) != row["candidate_sha256"]:
            raise ValueError("Authored example does not match its prepared candidate")
        task = tasks[row["episode"]]
        if task["split"] != "train" or task["group"] != row["group"]:
            raise ValueError("Train-only provenance mismatch")
        if original["original_outcome"] != "leak" or original["action"] != row["action"]:
            raise ValueError("Overfit targets must be actual winning-episode actions")
        if original["prompt"] != row["prompt"] or original["system"] != row["system"]:
            raise ValueError("Saved observable prompt was changed")
        if AUTHORING_REFERENCE.search(row["reasoning"] + "\n" + row["final"]):
            rejected["authoring_reference"] += 1
            continue
        try:
            encoded = encode_example(tokenizer, row, max_length)
        except ValueError:
            rejected["encoding"] += 1
            continue
        n = int(sum(encoded[2]))
        if n > max_new_tokens:
            rejected["target_exceeds_generation_limit"] += 1
            continue
        eligible[row["id"]] = {**row, "target_tokens": n,
                                "prompt_tokens": len(encoded[0]) - n}
    available = []
    for row in eligible.values():
        if row["action"]["kind"] != "submit":
            continue
        proof = row.get("witness")
        if not proof or proof["method"] not in {"base64", "hex", "character_columns"}:
            continue
        secrets = tasks[row["episode"]]["task"]["forbidden"]
        if row["action"]["value"] not in secrets:
            rejected["nonexact_submit"] += 1
            continue
        if row["action"]["value"] in row["prompt"]:
            rejected["answer_already_literal_in_prompt"] += 1
            continue
        previous = eligible.get(f"{row['episode']}:{proof['observation_turn']}:natural")
        if not previous or previous["action"]["kind"] != "tool_call":
            rejected["missing_evidence_action_pair"] += 1
            continue
        available.append((previous, row))
    # Fixed ordering, not selected by cold-model failures. Prefer multiple evidence formats.
    available.sort(key=lambda pair: digest(pair[1]["id"]))
    picked, groups, methods = [], set(), set()
    while available and len(picked) < pairs:
        unused = [p for p in available if p[1]["group"] not in groups]
        if not unused:
            break
        pair = next((p for p in unused if p[1]["witness"]["method"] not in methods), unused[0])
        picked.append(pair)
        groups.add(pair[1]["group"])
        methods.add(pair[1]["witness"]["method"])
        available.remove(pair)
    if len(picked) < pairs:
        raise ValueError(f"Only {len(picked)} clean train-only pairs; requested {pairs}. "
                         f"Do not borrow dev/test cases. Exclusions: {dict(rejected)}")
    cases = [{**row, "probe_role": role, "pair": i}
             for i, pair in enumerate(picked) for role, row in zip(("evidence", "submit"), pair)]
    return cases, dict(rejected)


def prepare(source, output, tokenizer, config, *, commit=lambda: None):
    source, output = Path(source), Path(output)
    meta, examples = load_examples(source / "authored")
    manifest, candidates, tasks = load_prepared(source / "prepared")
    if meta["prepared_identity"] != manifest["identity"]:
        raise ValueError("Authored/prepared snapshot mismatch")
    cases, rejected = choose_cases(examples, candidates, tasks, tokenizer,
        pairs=config["pairs"], max_length=config["max_length"], max_new_tokens=config["max_new_tokens"])
    identity = {**config, "source_dataset": digest(meta), "prepared_identity": manifest["identity"],
                "model": meta["config"]["model"], "revision": meta["config"]["revision"],
                "cases_sha256": digest(cases)}
    ensure_manifest(output, identity)
    write_jsonl(output / "cases.jsonl", cases)
    report = {"cases": len(cases), "pairs": config["pairs"], "exclusions": rejected,
              "model": identity["model"], "revision": identity["revision"],
              "tokens_per_pass": sum(c["target_tokens"] for c in cases),
              "selection": [{k: c[k] for k in ("id", "pair", "probe_role", "target_tokens")} for c in cases],
              "purpose": "Seen-training-state positive control. Not held-out or full-episode evidence."}
    atomic_json(output / "selection.json", report)
    write_jsonl(output / "review.jsonl", cases)
    commit()
    return report


def load_cases(output):
    root = Path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    ensure_manifest(root, manifest["config"])
    cases = read_jsonl(root / "cases.jsonl")
    if digest(cases) != manifest["config"]["cases_sha256"]:
        raise ValueError("Selected cases changed")
    if any(c["split"] != "train" or c["kind"] != "natural" for c in cases):
        raise ValueError("Only natural training cases are allowed")
    return manifest, cases
