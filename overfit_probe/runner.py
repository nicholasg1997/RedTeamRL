"""Cheap seen-state control: cold HF -> bounded fit -> HF + fresh vLLM checks.

Generated actions are scored, not executed. Exact action reproduction is deliberately stricter
than task success; this test cannot establish generalization or autonomous recovery.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path

from redteamrl.attack_sft.example import clean_action
from redteamrl.scratchpad.agent import split_notes_and_action
from redteamrl.scratchpad.experiment import atomic_json, ensure_manifest
from teacher_sft.data import digest
from teacher_sft.encoding import encode_example
from teacher_sft.train import DEFAULTS, evaluate_loss, latest_checkpoint, train_loop
from overfit_probe.data import load_cases

DEFAULT_CONFIG = dict(max_steps=40, max_target_tokens=80000, rank=16, learning_rate=1e-4,
                      seed=42, batch_size=4, max_length=8192, max_new_tokens=3072,
                      max_model_len=12288, workers=4)


def validate_config(config):
    if not 1 <= config["max_steps"] <= 100 or not 1 <= config["max_target_tokens"] <= 200000:
        raise ValueError("Probe bounds: max_steps 1-100 and max_target_tokens 1-200000")
    if config["rank"] not in (8, 16) or not 0 < config["learning_rate"] <= .001:
        raise ValueError("Probe requires rank 8/16 and learning_rate in (0, .001]")
    if not 1 <= config["workers"] <= 8:
        raise ValueError("Probe workers must be 1-8")


def messages_for(case):
    # Intentionally no case.action, reasoning, final, witness, or teacher hint here.
    return [{"role": "system", "content": case["system"]},
            {"role": "user", "content": case["prompt"]}]


def grade(case, response):
    notes, action = split_notes_and_action(response.get("content") or "", case["tools"])
    complete = response.get("finish_reason") == "stop"
    thinking = bool((response.get("reasoning") or "").strip())
    valid = not action.validation_error
    return {"id": case["id"], "pair": case["pair"], "role": case["probe_role"],
            "status": "completed", **response, "action": clean_action(action.model_dump()),
            "complete": complete, "valid_action": valid,
            "format_ok": complete and thinking and valid and notes.startswith("## "),
            "exact_action": complete and valid and clean_action(action.model_dump()) == case["action"]}


def probe_cases(cases, output, generate, *, identity, workers=1, commit=lambda: None):
    """generate(case) -> reasoning/content/finish metadata. Resume records by exact case identity."""
    output = Path(output)
    ensure_manifest(output, {"run_identity": identity, "case_digest": digest(cases)})
    records, pending = {}, []
    for case in cases:
        path = output / (digest(case["id"]) + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            if record["case_sha256"] != digest(case):
                raise ValueError("Probe resume case mismatch")
            records[case["id"]] = record
        else:
            pending.append(case)
    def one(case):
        try:
            record = grade(case, generate(case))
        except Exception as exc:
            record = {"id": case["id"], "pair": case["pair"], "role": case["probe_role"],
                      "status": "failed", "exact_action": False, "format_ok": False,
                      "error_type": type(exc).__name__, "error": str(exc)}
        return {**record, "case_sha256": digest(case)}
    # HF uses workers=1; only vLLM requests run concurrently. The coordinator alone commits.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, c): c for c in pending}
        for future in as_completed(futures):
            case, record = futures[future], future.result()
            atomic_json(output / (digest(case["id"]) + ".json"), record)
            commit()
            records[case["id"]] = record
            print(f"{output.name}: {case['id']} exact={record['exact_action']} "
                  f"format={record['format_ok']} status={record['status']}", flush=True)
    ordered = [records[c["id"]] for c in cases]
    summary = summarize(ordered)
    atomic_json(output / "summary.json", summary)
    commit()
    return ordered


def summarize(records):
    return dict(requested=len(records), completed=sum(r["status"] == "completed" for r in records),
                failed=sum(r["status"] != "completed" for r in records),
                length_finishes=sum(r.get("finish_reason") == "length" for r in records),
                non_stop_finishes=sum(r["status"] == "completed" and r.get("finish_reason") != "stop" for r in records),
                missing_reasoning=sum(r["status"] == "completed" and not r.get("reasoning") for r in records),
                exact_actions=sum(r["exact_action"] for r in records),
                format_ok=sum(r["format_ok"] for r in records),
                by_role={role: {"n": sum(r["role"] == role for r in records),
                    "exact": sum(r["role"] == role and r["exact_action"] for r in records)}
                    for role in ("evidence", "submit")})


def action_encoded(tokenizer, case, max_length):
    """Conditional JSON-only NLL, given the reference rationale. Not autonomous accuracy."""
    ids, labels, weights = encode_example(tokenizer, case, max_length)
    from redteamrl.scratchpad.agent import _last_json_span
    content = "<think>\n" + case["reasoning"] + "\n</think>\n\n" + case["final"]
    text = tokenizer.apply_chat_template(messages_for(case) + [{"role": "assistant", "content": content}],
        tokenize=False, add_generation_prompt=False, enable_thinking=True)
    span = _last_json_span(case["final"])
    if not span:
        raise ValueError("No reference JSON action")
    start = text.rindex(case["final"]) + span[0]
    end = text.rindex(case["final"]) + span[1]
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if offsets["input_ids"] != ids:
        raise ValueError("Action token alignment mismatch")
    labels = [label if b > start and a < end else -100
              for label, (a, b) in zip(labels, offsets["offset_mapping"])]
    weights = [float(label != -100) for label in labels]
    if not any(weights):
        raise ValueError("No supervised action tokens")
    return ids, labels, weights


def hf_generate(model, tokenizer, case, config):
    import torch
    prompt = tokenizer.apply_chat_template(messages_for(case), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True)
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    model.eval()
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    endings = set(eos if isinstance(eos, list) else [eos])
    with torch.inference_mode():
        tokens = model.generate(**batch, do_sample=False, temperature=None, top_p=None, top_k=None,
            max_new_tokens=config["max_new_tokens"], pad_token_id=tokenizer.pad_token_id,
            use_cache=True)[0, batch["input_ids"].shape[1]:]
    raw = tokenizer.decode(tokens, skip_special_tokens=True)
    reason, separator, final = raw.partition("</think>")
    return {"reasoning": reason.removeprefix("<think>").strip() if separator else "",
            "content": final.strip() if separator else raw,
            "raw_completion": raw, "completion_tokens": len(tokens),
            "finish_reason": "stop" if len(tokens) and int(tokens[-1]) in endings else "length"}


def parameter_digest(model):
    sha = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.requires_grad:
            sha.update(name.encode())
            sha.update(param.detach().float().cpu().numpy().tobytes())
    return sha.hexdigest()


def train_and_probe(root, config, *, commit=lambda: None):
    """One HF model in memory. Returns only after releasing its GPU allocation."""
    import gc
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from redteamrl.train.learner import prepare_for_long_context_training
    root = Path(root)
    manifest, cases = load_cases(root)
    config = {**DEFAULT_CONFIG, **config}
    validate_config(config)
    for key in ("max_length", "max_new_tokens"):
        if config[key] != manifest["config"][key]:
            raise ValueError(f"Selection/run {key} differs")
    config.update(model=manifest["config"]["model"], revision=manifest["config"]["revision"])
    run = ensure_manifest(root / "run", {**config, "selection_identity": manifest["identity"]})
    identity = run["identity"]
    report_path = root / "fit.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["run_identity"] != identity:
            raise ValueError("Completed fit identity mismatch")
        return report
    tok = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    tok.pad_token = tok.pad_token or tok.eos_token
    encoded = [encode_example(tok, c, config["max_length"]) for c in cases]
    action_targets = [action_encoded(tok, c, config["max_length"]) for c in cases]
    if any(c["prompt_tokens"] + config["max_new_tokens"] > config["max_model_len"] for c in cases):
        raise ValueError("Generation context exceeds the selected window")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = optimizer = None
    try:
        model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config["revision"],
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda"})
        ckpt = latest_checkpoint(root / "checkpoints")
        if ckpt:
            if json.loads((ckpt / "meta.json").read_text())["identity"] != identity:
                raise ValueError("Checkpoint identity mismatch")
            model = PeftModel.from_pretrained(model, ckpt, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(r=config["rank"], lora_alpha=2 * config["rank"],
                lora_dropout=0., bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        initial_path = root / "initial-adapter.json"
        if not initial_path.exists():
            if ckpt:
                raise ValueError("Initial adapter fingerprint missing on resume")
            atomic_json(initial_path, {"sha256": parameter_digest(model)})
            commit()
        with model.disable_adapter():
            before = dict(all_tokens=evaluate_loss(model, tok, encoded, "cuda"),
                          conditional_action=evaluate_loss(model, tok, action_targets, "cuda"))
            baseline = probe_cases(cases, root / "hf-base", lambda c: hf_generate(model, tok, c, config),
                                   identity=identity, commit=commit)
        if baseline_ceiling(baseline) and ckpt is None:
            initial = json.loads(initial_path.read_text())["sha256"]
            report = dict(run_identity=identity, config=config, step=0, presentations=0,
                target_tokens_seen=0, history=[], seen_loss_before=before, seen_loss_after=None,
                adapter=None, initial_adapter_sha256=initial, final_adapter_sha256=initial,
                parameters_changed=False, skipped_training=True, promoted=False)
            atomic_json(report_path, report)
            commit()
            return report
        prepare_for_long_context_training(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=config["learning_rate"], weight_decay=0., fused=True)
        state = None
        if ckpt:
            saved = torch.load(ckpt / "optimizer.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(saved["optimizer"])
            state = saved["state"]
            torch.set_rng_state(saved["rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
            del saved
        def save(model, optimizer, state):
            import shutil
            final = root / "checkpoints" / f"step-{state['step']:05d}"
            if (final / "meta.json").exists():
                return
            temp = final.parent / f".pending-{state['step']:05d}"
            if temp.exists():
                shutil.rmtree(temp)
            temp.mkdir(parents=True)
            model.save_pretrained(temp)
            tok.save_pretrained(temp)
            torch.save({"optimizer": optimizer.state_dict(), "state": state,
                        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temp / "optimizer.pt")
            atomic_json(temp / "meta.json", {"identity": identity, "step": state["step"]})
            temp.rename(final)
            commit()
        train_config = {**DEFAULTS, **config, "epochs": math.ceil(config["max_steps"] * config["batch_size"] / len(cases)),
                        "microbatch_size": 1, "checkpoint_every": 10, "alpha": 2 * config["rank"]}
        # This intentional overlap is the positive control. These losses are NOT held-out dev loss.
        state = train_loop(model, tok, encoded, encoded, optimizer, train_config,
                           state=state, checkpoint=save)
        after = dict(all_tokens=evaluate_loss(model, tok, encoded, "cuda"),
                     conditional_action=evaluate_loss(model, tok, action_targets, "cuda"))
        probe_cases(cases, root / "hf-tuned", lambda c: hf_generate(model, tok, c, config),
                    identity=identity, commit=commit)
        initial = json.loads(initial_path.read_text())["sha256"]
        final_digest = parameter_digest(model)
        report = dict(run_identity=identity, config=config, step=state["step"],
            presentations=state["cursor"], target_tokens_seen=state["target_tokens_seen"],
            history=state["history"], seen_loss_before=before, seen_loss_after=after,
            adapter=str(latest_checkpoint(root / "checkpoints")),
            initial_adapter_sha256=initial, final_adapter_sha256=final_digest,
            parameters_changed=initial != final_digest, promoted=False)
        atomic_json(report_path, report)
        commit()
        return report
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def served_probes(root, fit, *, commit=lambda: None):
    from redteamrl.policies.vllm_client import make_vllm_generate, start_vllm_server, stop_vllm_server
    root = Path(root)
    _, cases = load_cases(root)
    config = fit["config"]
    server = None
    try:
        # One fresh process loads the adapter at startup. No runtime swaps or reused adapter cache.
        skipped = fit.get("skipped_training", False)
        server = start_vllm_server(config["model"], 8000, .85, max_model_len=config["max_model_len"],
            max_num_seqs=config["workers"], lora_name=None if skipped else "overfit", lora_path=fit["adapter"],
            max_lora_rank=config["rank"], extra_args=["--revision", config["revision"],
                                                      "--reasoning-parser", "qwen3", "--enforce-eager"])
        for arm in (("base",) if skipped else ("base", "tuned")):
            def generate(case):
                events = []
                gen = make_vllm_generate("http://localhost:8000", "overfit" if arm == "tuned" else config["model"],
                    temperature=0., top_p=1., max_tokens=config["max_new_tokens"], seed=config["seed"],
                    enable_thinking=True, require_reasoning=True, on_response=events.append)
                try:
                    chat = messages_for(case)
                    gen(chat[0]["content"], chat[1:])
                except Exception:
                    if not events:
                        raise
                event = events[-1]
                return {"reasoning": event.get("reasoning"), "content": event.get("final_content"),
                        "finish_reason": event.get("finish_reason"), "generation": event}
            probe_cases(cases, root / f"vllm-{arm}", generate, identity=fit["run_identity"],
                        workers=config["workers"], commit=commit)
    finally:
        stop_vllm_server(server)


def baseline_ceiling(records):
    threshold = len(records) - 1
    return (len(records) >= 4 and all(r["status"] == "completed" and r.get("finish_reason") == "stop" for r in records)
            and sum(r["exact_action"] for r in records) >= threshold
            and sum(r["format_ok"] for r in records) >= threshold)


def decision_report(root):
    root = Path(root)
    fit = json.loads((root / "fit.json").read_text())
    arms = {}
    records = {}
    skipped = fit.get("skipped_training", False)
    for name in (("hf-base", "vllm-base") if skipped else ("hf-base", "hf-tuned", "vllm-base", "vllm-tuned")):
        arms[name] = json.loads((root / name / "summary.json").read_text())
        records[name] = {r["id"]: r for p in (root / name).glob("*.json")
                         if "case_sha256" in (r := json.loads(p.read_text()))}
    n = arms["hf-base"]["requested"]
    threshold = n - 1
    def gain(backend):
        a, b = records[f"{backend}-base"], records[f"{backend}-tuned"]
        return {"gained": sum(not a[k]["exact_action"] and b[k]["exact_action"] for k in a),
                "lost": sum(a[k]["exact_action"] and not b[k]["exact_action"] for k in a)}
    gains = {} if skipped else {b: gain(b) for b in ("hf", "vllm")}
    issues = any(s["failed"] or s["non_stop_finishes"] or s["missing_reasoning"] for s in arms.values())
    if issues:
        verdict = "INCONCLUSIVE_GENERATION_ERRORS"
    elif skipped:
        verdict = "BASELINE_ALREADY_SOLVES" if baseline_ceiling(list(records["vllm-base"].values())) else "INVESTIGATE_SERVING_DIFFERENCE"
    elif not fit["parameters_changed"]:
        verdict = "FAIL_NO_PARAMETER_CHANGE"
    elif all(arms[f"{b}-base"]["exact_actions"] >= threshold for b in ("hf", "vllm")):
        verdict = "BASELINE_ALREADY_SOLVES"
    elif arms["hf-tuned"]["exact_actions"] >= threshold and arms["vllm-tuned"]["exact_actions"] < threshold:
        verdict = "INVESTIGATE_SERVING_DIFFERENCE"
    elif all(arms[f"{b}-tuned"]["exact_actions"] >= threshold and
             arms[f"{b}-tuned"]["format_ok"] >= threshold and gains[b]["gained"] >= 2 and gains[b]["lost"] == 0
             for b in ("hf", "vllm")):
        verdict = "PASS_SEEN_STATE_LEARNING"
    else:
        verdict = "NOT_DEMONSTRATED_AT_THIS_BUDGET"
    result = {"verdict": verdict, "threshold": threshold, "cases": n, "arms": arms,
              "paired_changes": gains, "parameters_changed": fit["parameters_changed"],
              "seen_loss_before": fit["seen_loss_before"], "seen_loss_after": fit["seen_loss_after"],
              "step": fit["step"], "presentations": fit["presentations"],
              "target_tokens_seen": fit["target_tokens_seen"], "skipped_training": skipped, "promoted": False,
              "purpose": "Deliberate memorization/serving control; no autonomous win or generalization claim.",
              "caveat": "Equivalent alternative actions fail exact reproduction; inspect them before diagnosing inability."}
    atomic_json(root / "report.json", result)
    return result
