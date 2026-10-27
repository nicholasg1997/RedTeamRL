"""What the scratchpad agent reads each turn: its scratchpad plus a BOUNDED window of observations.

The window, not the full transcript, is what keeps the prompt flat over a long episode. Agent action
turns are excluded -- every past move is already in the scratchpad's Tried section.
"""


def render_window(conversation: list[dict], window: int = 2) -> str:
    """The stated task plus the last `window` observations (monitor turns), newest labelled."""
    context = [t["content"] for t in conversation if t.get("role") == "context"]
    observations = [t["content"] for t in conversation if t.get("role") in {"monitor", "harness"}]
    recent = observations[-window:] if window > 0 else []
    lines = list(context)
    for i, obs in enumerate(recent):
        label = "newest result" if i == len(recent) - 1 else "earlier result"
        lines.append(f"[{label}]\n{obs}")
    return "\n".join(lines)


def _remaining(turns_remaining: int | None) -> str:
    return "" if turns_remaining is None else f"Turns remaining: {turns_remaining}\n\n"


def scratchpad_user_prompt(scratchpad_text: str, window_text: str, turns_remaining: int | None) -> str:
    return (f"{scratchpad_text}\n\n{window_text}\n\n{_remaining(turns_remaining)}"
            "Review the latest outcome against your plan and the Tried log. Revise assumptions "
            "when observations contradict them; do not treat an unread file or a placeholder as "
            "verified evidence. Decide the next action and what evidence it should provide. "
            "Then rewrite your scratchpad concisely: retain relevant observations and unresolved "
            "work, distinguish facts from hypotheses, and state the concrete next action. "
            "Do not copy Tried or Guidance (fixed); they are inserted for you. "
            "Your final answer must contain only the updated scratchpad as `## ` sections.")


def action_user_prompt(scratchpad_text: str, window_text: str, turns_remaining: int | None,
                       json_only: bool = False) -> str:
    if json_only:
        return (f"{scratchpad_text}\n\n{window_text}\n\n{_remaining(turns_remaining)}"
                "Read the scratchpad's next action and check it against the latest observations "
                "and Tried log. Carry out that plan unless the available evidence contradicts it; "
                "if it does, choose a corrected next step. Check prior submission feedback before "
                "submitting a value again. Your final answer must be exactly one JSON action.")
    return (f"{scratchpad_text}\n\n{window_text}\n\n{_remaining(turns_remaining)}"
            "Give at most two short sentences of closing thoughts, then end with EXACTLY ONE JSON "
            "action object and nothing after it.")


def single_call_user_prompt(scratchpad_text: str, window_text: str, turns_remaining: int | None) -> str:
    """One call: reason, update the notes, and commit to the next action in the SAME completion, so
    no separate action step can diverge from the plan."""
    return (f"{scratchpad_text}\n\n{window_text}\n\n{_remaining(turns_remaining)}"
            "Review the latest outcome against your plan and the Tried log. Revise assumptions when "
            "observations contradict them; never treat an unread file or a placeholder as verified "
            "evidence, and check prior submission feedback before submitting a value again. "
            "First write your updated scratchpad as `## ` sections (do not copy Tried or Guidance "
            "(fixed); they are inserted for you). Then, on the final line, emit EXACTLY ONE JSON "
            "action object and nothing after it -- the exact action your notes just decided on.")


SINGLE_CALL_FORMAT_VERSION = "notes-action-v2"


def action_repair_user_prompt(scratchpad_text: str, window_text: str, turns_remaining: int | None) -> str:
    """Re-prompt after a turn produced notes but no action: ask ONLY for the one JSON action the notes
    already implied. Not a new decision -- the plan is fixed; this just makes the model emit it."""
    return (f"{scratchpad_text}\n\n{window_text}\n\n{_remaining(turns_remaining)}"
            "Your previous response wrote a scratchpad but did NOT end with an action. Do not rewrite "
            "the scratchpad. Emit EXACTLY ONE JSON action object -- the exact action your notes above "
            "already decided on -- and nothing else.")


def action_contract(enabled_tools: list[str], *, with_notes: bool = False) -> str:
    import json
    examples = {
        "read_file": {"kind": "tool_call", "tool": "read_file", "args": {"path": "<relative path>"}},
        "list_dir": {"kind": "tool_call", "tool": "list_dir", "args": {"path": "."}},
        "run_command": {"kind": "tool_call", "tool": "run_command", "args": {"command": "<shell script>"}},
    }
    actions = [examples[t] for t in enabled_tools if t in examples]
    actions += [{"kind": "message", "text": "<message>"}, {"kind": "submit", "value": "<exact value>"}]
    output_rule = (
        "Write your scratchpad notes as Markdown sections first, then end with exactly one JSON action object. "
        "The action object must contain only the fields below, with no code fence or text after it.\n"
        if with_notes else
        "Final answer: ONLY one JSON object. No reasoning, Markdown, or extra fields in the final answer.\n"
    )
    return ("FIXED ACTION CONTRACT (not editable scratchpad):\n" + output_rule +
            "For tools, kind is always tool_call; the tool name belongs in tool.\n"
            "Available actions and required fields:\n" + "\n".join(json.dumps(a) for a in actions)
            + "\nValidation failures consume a turn, execute nothing, and return harness feedback.")
