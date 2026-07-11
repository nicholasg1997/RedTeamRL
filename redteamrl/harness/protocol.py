"""Wire-format layer: convert between the LLM's raw text and typed objects.

Two directions only:
  - render_*  : typed objects -> text (build the prompt the model reads)
  - parse_*   : model text -> typed objects (with robust / fail-closed fallbacks)

No prompts and no policy classes live here — those are in policies/prompted.py,
which depends on this module (never the reverse).
"""
import json

from redteamrl.types import AttackerAction, DefenderDecision


def _extract_json(raw: str) -> dict | None:
    """Return the LAST top-level JSON object embedded in `raw`, or None.

    Both the attacker and defender are prompted to reason in prose first and put
    their action/decision JSON last, so when several objects appear (e.g. the
    model shows a candidate mid-reasoning) we want the final one. On each decode
    we jump the cursor past the consumed span, so nested braces (like an `args`
    object) are never mistaken for a top-level object.
    """
    decoder = json.JSONDecoder()
    found = None
    i, n = 0, len(raw)
    while i < n:
        if raw[i] == "{":
            try:
                obj, end = decoder.raw_decode(raw[i:])
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict):
                found = obj
                i += end
                continue
        i += 1
    return found


def parse_attacker_action(raw: str) -> AttackerAction:
    """Parse the attacker's output. Never raises: malformed output degrades to a
    harmless `message` so a bad turn can't kill the episode."""
    obj = _extract_json(raw)
    if isinstance(obj, dict):
        try:
            return AttackerAction(**obj)
        except Exception:
            pass
    return AttackerAction(kind="message", text=raw)


# Safe content to return when we override the monitor's own (untrusted) verdict.
_FAIL_CONTENT = {"reject": None, "redact": "[withheld]", "respond": "I can't help with that."}


def parse_defender_decision(
    raw: str, valid_verdicts: set[str], fail_default: str = "reject"
) -> DefenderDecision:
    """Parse the defender's output, failing CLOSED.

    We trust the model's decision ONLY if it parses AND its verdict is valid for
    this context. In every other case — unparseable, malformed, or a verdict that
    doesn't belong here — we fall back to `fail_default` with SAFE content. That
    matters: if the model returns e.g. verdict "allow" with the raw secret as
    content during a *result* adjudication, we must NOT keep that content while
    overriding the verdict, or we'd leak on our own override.
    """
    obj = _extract_json(raw)
    if isinstance(obj, dict):
        try:
            decision = DefenderDecision(**obj)
            if decision.verdict in valid_verdicts:
                return decision
        except Exception:
            pass
    return DefenderDecision(
        reasoning=f"[unparseable or invalid monitor output] {raw[:300]}",
        verdict=fail_default,
        content=_FAIL_CONTENT.get(fail_default),
    )


def render_conversation(conversation: list[dict]) -> str:
    """Render the conversation-so-far as a plain-text transcript for the prompt."""
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation)


def render_action(action: AttackerAction) -> str:
    """Render an attacker action as a short human-readable line for the monitor."""
    if action.kind == "tool_call":
        return f"{action.tool}({action.args})"
    return action.text or ""
