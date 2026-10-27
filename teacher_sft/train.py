"""Bounded LoRA SFT, with complete thinking targets and resumable optimizer state."""
import json
import math
from pathlib import Path
import random

from redteamrl.scratchpad.experiment import atomic_json, ensure_manifest
from teacher_sft.author import load_examples
from teacher_sft.data import digest
from teacher_sft.encoding import encode_example, select_arm

DEFAULTS = dict(arm="wins", seed=42, epochs=1, max_steps=20, batch_size=4,
                microbatch_size=1, max_length=8192, learning_rate=3e-5,
                rank=8, alpha=16, checkpoint_every=5, dev_limit=16, max_target_tokens=60000)


def validate_config(config):
    for key in ("epochs", "max_steps", "batch_size", "microbatch_size", "max_length", "rank",
                "alpha", "checkpoint_every", "dev_limit", "max_target_tokens"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive and finite")
    select_arm([], config["arm"])


def batch_loss(model, tokenizer, examples, device):
    from redteamrl.sft.sft_train import _pad_microbatch, _weighted_causal_loss
    ids, labels, mask, weights = _pad_microbatch(tokenizer, examples, device)
    # Qwen3 can omit vocabulary logits for the masked prompt. Keep one preceding position
    # so the first target token still has its causal prediction; this materially reduces memory.
    if getattr(getattr(model, "config", None), "model_type", None) == "qwen3":
        start = min(next(i for i, label in enumerate(row[1]) if label != -100) for row in examples)
        start = max(0, start - 1)
        output = model(input_ids=ids, attention_mask=mask, logits_to_keep=ids.shape[1] - start)
        labels, weights = labels[:, start:], weights[:, start:]
    else:
        output = model(input_ids=ids, attention_mask=mask)
    return _weighted_causal_loss(output, labels, weights)


def target_tokens(rows):
    return sum(sum(w for label, w in zip(labels[1:], weights[1:]) if label != -100)
               for _, labels, weights in rows)


def evaluate_loss(model, tokenizer, examples, device):
    import torch
    if not examples:
        raise ValueError("A nonempty dev set is required")
    model.eval()
    total, tokens = 0., 0
    with torch.no_grad():
        for row in examples:
            n = target_tokens([row])
            value = float(batch_loss(model, tokenizer, [row], device))
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite dev loss")
            total += value * n
            tokens += n
    return total / tokens


def train_loop(model, tokenizer, train, dev, optimizer, config, *, device="cuda", state=None,
               checkpoint=lambda model, optimizer, state: None):
    """Microbatches are weighted by actual supervised tokens, including a partial final batch."""
    import torch
    validate_config(config)
    if not train or not dev:
        raise ValueError("Both train and dev examples are required; collect/author more if empty")
    order = []
    rng = random.Random(config["seed"])
    for _ in range(config["epochs"]):
        indices = list(range(len(train)))
        rng.shuffle(indices)
        order.extend(indices)
    state = dict(state) if state is not None else {
        "step": 0, "cursor": 0, "history": [],
        "target_tokens_seen": 0,
        "dev_loss_before": evaluate_loss(model, tokenizer, dev, device)}
    while state["step"] < config["max_steps"] and state["cursor"] < len(order):
        indices = order[state["cursor"]:state["cursor"] + config["batch_size"]]
        while indices and state["target_tokens_seen"] + target_tokens([train[i] for i in indices]) > config["max_target_tokens"]:
            indices.pop()
        if not indices:
            break
        batch = [train[i] for i in indices]
        total_tokens = target_tokens(batch)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        value = 0.
        for offset in range(0, len(batch), config["microbatch_size"]):
            micro = batch[offset:offset + config["microbatch_size"]]
            weight = target_tokens(micro) / total_tokens
            loss = batch_loss(model, tokenizer, micro, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; checkpoint not advanced")
            (loss * weight).backward()
            value += float(loss.detach()) * weight
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        state["cursor"] += len(batch)
        state["step"] += 1
        state["target_tokens_seen"] += int(total_tokens)
        state["history"].append(dict(step=state["step"], loss=value, gradient_norm=float(norm),
                                     target_tokens=int(total_tokens)))
        print(f"SFT step={state['step']} loss={value:.4f} grad={float(norm):.4f} "
              f"target_tokens={int(total_tokens)}", flush=True)
        if state["step"] % config["checkpoint_every"] == 0:
            checkpoint(model, optimizer, state)
    state["dev_loss_after"] = evaluate_loss(model, tokenizer, dev, device)
    checkpoint(model, optimizer, state)
    return state


def latest_checkpoint(output):
    paths = list(Path(output).glob("step-*/meta.json"))
    return max(paths, key=lambda p: int(p.parent.name.split("-")[1])).parent if paths else None


def train_adapter(dataset, output, config, *, commit=lambda: None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from redteamrl.train.learner import prepare_for_long_context_training

    config = {**DEFAULTS, **config}
    validate_config(config)
    meta, examples = load_examples(dataset)
    for key in ("model", "revision", "source"):
        if not config.get(key) or config[key] != meta["config"].get(key):
            raise ValueError(f"Student author/training {key} mismatch")
    output = Path(output)
    identity = {**config, "dataset_sha256": digest(meta), "format": "native-thinking-v1"}
    manifest = ensure_manifest(output, identity)
    commit()
    tok = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    tok.pad_token = tok.pad_token or tok.eos_token
    selected = select_arm(examples, config["arm"])
    encoded, rejected = {"train": [], "dev": []}, []
    for row in sorted(selected, key=lambda r: digest(r["id"])):
        try:
            encoded[row["split"]].append(encode_example(tok, row, config["max_length"]))
        except ValueError as exc:
            rejected.append({"id": row["id"], "reason": str(exc)})
    encoded["dev"] = encoded["dev"][:config["dev_limit"]]
    atomic_json(output / "encoding.json", {"accepted": {k: len(v) for k, v in encoded.items()},
                                           "rejected": rejected})
    commit()
    if not encoded["train"] or not encoded["dev"]:
        raise ValueError("No usable train/dev set; inspect encoding.json before authoring more")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config["revision"],
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda"})
    checkpoint_path = latest_checkpoint(output)
    if checkpoint_path:
        saved_meta = json.loads((checkpoint_path / "meta.json").read_text())
        if saved_meta["identity"] != manifest["identity"]:
            raise ValueError("Checkpoint identity mismatch")
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=config["rank"], lora_alpha=config["alpha"],
            lora_dropout=0., bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    prepare_for_long_context_training(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["learning_rate"], weight_decay=0., fused=True)
    state = None
    if checkpoint_path:
        saved = torch.load(checkpoint_path / "optimizer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        state = saved["state"]

    def save(model, optimizer, state):
        import shutil
        final = output / f"step-{state['step']:05d}"
        # Never overwrite a complete checkpoint. A final dev report lives separately below.
        if (final / "meta.json").exists():
            return
        temp = output / f".pending-{state['step']:05d}"
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        model.save_pretrained(temp)
        tok.save_pretrained(temp)
        torch.save({"optimizer": optimizer.state_dict(), "state": state,
                    "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
                   temp / "optimizer.pt")
        atomic_json(temp / "meta.json", {"identity": manifest["identity"], "step": state["step"],
                                        "model": config["model"], "revision": config["revision"]})
        temp.rename(final)
        commit()

    state = train_loop(model, tok, encoded["train"], encoded["dev"], optimizer, config,
                       state=state, checkpoint=save)
    report = {**state, "identity": manifest["identity"], "arm": config["arm"],
              "adapter": str(latest_checkpoint(output)), "promoted": False,
              "note": "Teacher-forced loss is diagnostic; paired behavioral evaluation is required."}
    atomic_json(output / "report.json", report)
    commit()
    return report
