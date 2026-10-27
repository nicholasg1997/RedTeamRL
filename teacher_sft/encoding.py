"""Native thinking-on chat template; prompt/padding masked, complete targets only."""


def encode_example(tokenizer, row, max_length=8192):
    if not row["reasoning"].strip() or not row["final"].strip():
        raise ValueError("Missing reasoning or final target")
    chat = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["prompt"]}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=True)
    content = "<think>\n" + row["reasoning"] + "\n</think>\n\n" + row["final"]
    full = tokenizer.apply_chat_template(chat + [{"role": "assistant", "content": content}],
                                          tokenize=False, add_generation_prompt=False, enable_thinking=True)
    if not full.startswith(prompt):
        raise ValueError("Tokenizer assistant prefix differs from inference prefix")
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer boundary mismatch; cannot safely mask prompt")
    if full.count("<think>") - prompt.count("<think>") not in (0, 1) or "</think>" not in full[len(prompt):]:
        raise ValueError("Tokenizer dropped or duplicated reasoning markers")
    if len(ids) > max_length:
        raise ValueError(f"Example {row['id']} is {len(ids)} tokens; refusing target truncation")
    if len(prompt_ids) >= len(ids):
        raise ValueError("No supervised target tokens")
    n = len(prompt_ids)
    return ids, [-100] * n + ids[n:], [0.0] * n + [1.0] * (len(ids) - n)


def select_arm(examples, arm):
    if arm not in {"wins", "wins+corrections"}:
        raise ValueError("arm must be wins or wins+corrections")
    return [r for r in examples if arm == "wins+corrections" or r["kind"] == "natural"]
