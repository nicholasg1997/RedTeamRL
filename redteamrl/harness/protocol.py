"""Wire-format layer: convert between the LLM's raw text and typed objects.

Two directions only:
  - render_*  : typed objects -> text (build the prompt the model reads)
  - parse_*   : model text -> typed objects (with robust / fail-closed fallbacks)

No prompts and no policy classes live here — those are in policies/prompted.py,
which depends on this module (never the reverse).
"""
import json

from redteamrl.types import AttackerAction, DefenderDecision

_INVALID_ATTACKER_OUTPUT = "[invalid attacker output: expected one final JSON action]"
_DEFENDER_FAIL_MARKER = "[unparseable or invalid monitor output]"   # prefix of a fail-closed decision's reasoning
_OMITTED_TRANSCRIPT = "[... earlier conversation omitted to fit context budget ...]"
_TRUNCATED_TURN = "[... turn content truncated to fit context budget ...]"


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
    harmless `message` so a bad turn can't kill the episode.

    Do not put `raw` in the fallback message. With thinking-enabled models, raw
    may contain thousands of tokens of private reasoning; if that text becomes a
    visible message it is appended to the episode transcript and explodes future
    prompts. We DO keep it in the CoT-safe `raw` field (never rendered to the
    monitor — see render_action) so a failed parse stays diagnosable after the run:
    truncated thinking, empty content, and malformed JSON look identical in the
    transcript otherwise.
    """
    obj = _extract_json(raw)
    if isinstance(obj, dict):
        try:
            return AttackerAction(**obj)
        except Exception:
            pass
    return AttackerAction(kind="message", text=_INVALID_ATTACKER_OUTPUT, raw=raw)


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
        reasoning=f"{_DEFENDER_FAIL_MARKER} {raw[:300]}",
        verdict=fail_default,
        content=_FAIL_CONTENT.get(fail_default),
    )


def _render_turn(turn: dict) -> str:
    return f"{turn['role']}: {turn['content']}"


def _truncate_turn(turn: dict, max_chars: int) -> str:
    prefix = f"{turn['role']}: {_TRUNCATED_TURN}\n"
    if max_chars <= len(prefix):
        return prefix[:max_chars]
    content = str(turn["content"])
    return prefix + content[-(max_chars - len(prefix)) :]


def render_conversation(conversation: list[dict], max_chars: int | None = None) -> str:
    """Render the conversation-so-far as a plain-text transcript for the prompt."""
    rendered = "\n".join(_render_turn(turn) for turn in conversation)
    if max_chars is None or len(rendered) <= max_chars:
        return rendered

    if max_chars <= 0:
        return ""

    context_turns = [turn for turn in conversation if turn.get("role") == "context"]
    history_turns = [turn for turn in conversation if turn.get("role") != "context"]

    kept: list[str] = []
    used_chars = 0
    for turn in reversed(history_turns):
        rendered_turn = _render_turn(turn)
        separator_chars = 1 if kept else 0
        required_chars = len(rendered_turn) + separator_chars
        if used_chars + required_chars <= max_chars:
            kept.insert(0, rendered_turn)
            used_chars += required_chars
            continue

        remaining_chars = max_chars - used_chars - separator_chars
        if remaining_chars > 0 and not kept:
            kept.insert(0, _truncate_turn(turn, remaining_chars))
        break

    prefix_turns = [_render_turn(turn) for turn in context_turns]
    omitted = _OMITTED_TRANSCRIPT
    candidate_parts = prefix_turns + [omitted] + kept
    candidate = "\n".join(part for part in candidate_parts if part)
    if len(candidate) <= max_chars:
        return candidate

    tail_parts = [omitted] + kept
    tail = "\n".join(part for part in tail_parts if part)
    if len(tail) <= max_chars:
        return tail

    marker = omitted + "\n"
    if max_chars <= len(marker):
        return omitted[:max_chars]
    kept_tail = "\n".join(kept)
    return marker + kept_tail[-(max_chars - len(marker)) :]


def render_action(action: AttackerAction) -> str:
    """Render an attacker action as a short human-readable line for the monitor."""
    if action.kind == "tool_call":
        return f"{action.tool}({action.args})"
    return action.text or ""
