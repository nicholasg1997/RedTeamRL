"""The Tried log: one plain `action -> response` line per completed move.

Built from the agent's own prior actions and the monitor's returned observations, never written
by the model. A submit's value is the attacker's own private data, so it may appear in its Tried.
"""
from redteamrl.types import AttackerAction

_TRUNCATED = "..."


def summarize_action(action: AttackerAction, max_chars: int = 160) -> str:
    """The move, one line. Its detail is capped like a response, so a long command or message
    repeated over many turns cannot grow the prompt without bound."""
    if action.validation_error:
        return "invalid action: " + summarize_response(action.validation_error, max_chars)
    if action.kind == "tool_call":
        args = action.args or {}
        detail = summarize_response(str(args.get("path") or args.get("command") or ""), max_chars)
        return f"{action.tool} {detail}".rstrip()
    if action.kind == "message":
        return f'message "{summarize_response(action.text or "", max_chars)}"'
    return f'submit "{summarize_response(action.value or "", max_chars)}"'


def summarize_response(returned: str, max_chars: int) -> str:
    one_line = " ".join(returned.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[:max_chars] + _TRUNCATED


def tried_lines(actions: list[AttackerAction], responses: list[str], max_chars: int = 160) -> list[str]:
    """Pair each action with its response; an action with no response yet (the newest) is omitted."""
    return [f"{summarize_action(a, max_chars)} -> {summarize_response(r, max_chars)}"
            for a, r in zip(actions, responses)]
