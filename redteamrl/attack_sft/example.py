import json
from redteamrl.sft.filter import is_grounded


def clean_action(action: dict) -> dict:
    """Keep only wire-format AttackerAction fields, stripping internal harness metadata.

    Strips extra fields like 'substitutions', 'monitor', 'sensitive', 'monitor_response' that leak from
    raw episode step records into SFT target JSONs.
    """
    if not isinstance(action, dict):
        return {}
    kind = action.get("kind")
    if kind == "tool_call":
        out = {"kind": "tool_call", "tool": action.get("tool")}
        if "args" in action and action["args"] is not None:
            out["args"] = action["args"]
        return out
    if kind == "message":
        return {"kind": "message", "text": action.get("text") or ""}
    if kind == "submit":
        return {"kind": "submit", "value": action.get("value") or ""}
    out = {}
    for key in ("kind", "tool", "args", "text", "value"):
        if key in action and action[key] is not None:
            out[key] = action[key]
    return out


def render_target(reasoning: str, action: dict) -> str:
    """Render the attacker's native format: reasoning prose followed by JSON action.

    The PromptedAttacker parses by taking the LAST top-level JSON object,
    so this format is re-parseable by the same path. Action is sanitized to strip internal
    harness step metadata.
    """
    cleaned = clean_action(action)
    return f"{reasoning}\n{json.dumps(cleaned)}"


def scrub_secret(text: str, canary: str) -> str:
    """Replace every occurrence of canary with <the secret>.

    Scrubs the key from reasoning so SFT tunes strategy, not key-values.
    """
    return text.replace(canary, "<the secret>")


def build_example(
    system: str, observable_prompt: str, reasoning: str, action: dict, canary: str,
    train_action: bool = True
) -> dict | None:
    """Build an SFT example if reasoning is grounded, else drop it.

    Grounds the ORIGINAL (pre-scrub) reasoning against the canary: the exact canary may be cited
    only if this turn's observable prompt already contained it, and -- via ``authoring_values`` --
    so must every >=MIN_SECRET_FRAGMENT_CHARS fragment of it. That fragment scan is what catches a
    partial quote of the key that the exact-string scrub below would otherwise leave in the target;
    it is safe from false positives because episode canaries are high-entropy tokens with no
    semantic substrings (`sft.canary.episode_canary`), so ordinary technique language
    ("base64-encoded", "multi-part") is never mistaken for a secret fragment. The (unscrubbed)
    action is kept as-is, so submit actions preserve the real canary value.

    Returns {"system", "prompt", "target", "action_loss_weight"} if grounded, else None.
    action_loss_weight is 1.0 if train_action else 0.0 (for masked actions).
    """
    if not is_grounded(reasoning, [canary], observable_prompt, authoring_values=[canary]):
        return None

    # Scrub the canary from reasoning for the SFT example
    scrubbed_reasoning = scrub_secret(reasoning, canary)

    return {
        "system": system,
        "prompt": observable_prompt,
        "target": render_target(scrubbed_reasoning, action),
        "action_loss_weight": 1.0 if train_action else 0.0,
    }
