"""The scratchpad attacker: one `act()` per turn, two model calls inside it.

Generation 1 rewrites the scratchpad under its own token budget; generation 2 writes the
JSON action under a separate budget (legacy unconstrained mode also accepts closing thoughts). Either generation can still hit its own
limit, so the PoC records termination reasons. The budgets use two `generate` callables (each
pre-bound to its max_tokens) because the shared `generate(system, messages)` signature has no
per-call budget. The harness loop is unchanged: this class is just an `AttackerPolicy`.

Private state (scratchpad, thoughts, Tried, window) lives here and in `turn_records`. It never
enters the shared `conversation`, which the defender renders.
"""
import json
from dataclasses import dataclass

from redteamrl.harness.protocol import parse_attacker_action
from redteamrl.scratchpad.document import parse_scratchpad, render_scratchpad, seed_scratchpad
from redteamrl.scratchpad.render import (
    action_contract, action_repair_user_prompt, action_user_prompt, render_window,
    scratchpad_user_prompt, single_call_user_prompt,
)
from redteamrl.scratchpad.tried import tried_lines
from redteamrl.types import AttackerAction


@dataclass
class ScratchpadTurn:
    """Everything one turn produced, for the research record and (later) RL scoring."""
    scratchpad_text: str          # the rewritten scratchpad as rendered (Tried injected)
    closing_thoughts: str
    scratchpad_prompt: str
    scratchpad_completion: str
    action_prompt: str
    action_completion: str
    action: AttackerAction
    proposed_action: AttackerAction | None = None
    repaired: bool = False


def parse_action_generation(raw: str) -> tuple[str, AttackerAction]:
    """(closing thoughts, action). The thoughts are the text before the LAST top-level JSON object;
    the action goes through the fail-closed `parse_attacker_action`."""
    decoder = json.JSONDecoder()
    start, i = None, 0
    while i < len(raw):
        if raw[i] == "{":
            try:
                obj, end = decoder.raw_decode(raw[i:])
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict):
                start = i
            i += end
            continue
        i += 1
    thoughts = raw if start is None else raw[:start]
    return thoughts.strip(), parse_attacker_action(raw)


def _last_json_span(raw: str) -> tuple[int, int] | None:
    """(start, end) of the LAST top-level JSON object in `raw`, or None."""
    decoder = json.JSONDecoder()
    span, i = None, 0
    while i < len(raw):
        if raw[i] == "{":
            try:
                obj, end = decoder.raw_decode(raw[i:])
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict):
                span = (i, i + end)
            i += end
            continue
        i += 1
    return span


def split_notes_and_action(raw: str, enabled_tools: list[str]) -> tuple[str, AttackerAction]:
    """Single-call parse: notes are the text before the final JSON object; that object is the action,
    schema-validated (an invalid or absent action becomes a validation_error so the turn is consumed
    with harness feedback rather than silently executing something else)."""
    from redteamrl.harness.protocol import validate_action
    span = _last_json_span(raw)
    notes = raw[:span[0]].strip() if span else raw.strip()
    action = validate_action(raw[span[0]:span[1]] if span else "", enabled_tools)
    return notes, action


class ScratchpadAttacker:
    def __init__(self, scratchpad_generate, action_generate, *, system: str, goal: str, alias: str,
                 seed_next_ideas: str, window: int = 2, turn_budget: int | None = None,
                 rewrite_scratchpad: bool = True, constrained_actions: bool = False,
                 single_call: bool = False, enabled_tools: list[str] | None = None):
        self.scratchpad_generate = scratchpad_generate
        self.action_generate = action_generate
        self.constrained_actions = constrained_actions
        self.enabled_tools = enabled_tools if enabled_tools is not None else ["read_file", "list_dir", "run_command"]
        self.system = system
        self.goal = goal
        self.alias = alias
        self.window = window
        self.turn_budget = turn_budget
        self.rewrite_scratchpad = rewrite_scratchpad
        self.single_call = single_call
        self._guidance = seed_next_ideas               # pinned heuristics, re-injected every turn
        self._scratchpad = seed_scratchpad(goal, alias, "")   # episode-scoped; model planning starts blank
        self._actions: list[AttackerAction] = []
        self.turn_records: list[ScratchpadTurn] = []

    def act(self, conversation, turns_taken: int = 0) -> AttackerAction:
        observations = [t["content"] for t in conversation if t.get("role") in {"monitor", "harness"}]
        # Pair our own actions with the LAST len(actions) observations: a restart/prefix replay may
        # hand us a conversation that already holds turns we did not play.
        own = observations[len(observations) - len(self._actions):] if self._actions else []
        tried = tried_lines(self._actions, own)
        window_text = render_window(conversation, self.window)
        remaining = None if self.turn_budget is None else self.turn_budget - turns_taken

        if self.single_call:
            # ONE call decides the notes AND the action together, so there is no separate action
            # step that can diverge from the plan. The action-generate callable is unused.
            sp_text = render_scratchpad(self._scratchpad, tried, self._guidance)
            prompt = single_call_user_prompt(sp_text, window_text, remaining) + "\n\n" \
                + action_contract(self.enabled_tools, with_notes=True)
            raw = self.scratchpad_generate(self.system, [{"role": "user", "content": prompt}])
            notes, action = split_notes_and_action(raw, self.enabled_tools)
            rewritten = parse_scratchpad(notes)
            if rewritten.sections:
                self._scratchpad = rewritten
            sp_text = render_scratchpad(self._scratchpad, tried, self._guidance)

            # Action-repair: the completion had notes but no usable action. Re-prompt ONCE for just the
            # action the notes already implied (not a new decision, so no handoff divergence). One shot.
            repair_prompt, repair_raw, repaired = "", "", False
            if action.validation_error is not None:
                repair_prompt = (action_repair_user_prompt(sp_text, window_text, remaining)
                                 + "\n\n" + action_contract(self.enabled_tools))
                repair_raw = self.scratchpad_generate(self.system, [{"role": "user", "content": repair_prompt}])
                # Parse the repair like the primary path (tolerant last-JSON extraction), so a teacher
                # that wraps the action in prose/fence still repairs instead of wasting the call.
                _, repaired_action = split_notes_and_action(repair_raw, self.enabled_tools)
                if repaired_action.validation_error is None:
                    action, repaired = repaired_action, True

            self._actions.append(action)
            self.turn_records.append(ScratchpadTurn(
                sp_text, "", prompt, raw, repair_prompt, repair_raw, action,
                proposed_action=action, repaired=repaired))
            return action

        sp_prompt, sp_raw = "", ""
        if self.rewrite_scratchpad:
            sp_prompt = scratchpad_user_prompt(render_scratchpad(self._scratchpad, tried, self._guidance),
                                               window_text, remaining)
            sp_raw = self.scratchpad_generate(self.system, [{"role": "user", "content": sp_prompt}])
            rewritten = parse_scratchpad(sp_raw)
            if rewritten.sections:      # a headingless rewrite keeps the previous scratchpad
                self._scratchpad = rewritten
        sp_text = render_scratchpad(self._scratchpad, tried, self._guidance)

        act_prompt = action_user_prompt(sp_text, window_text, remaining, json_only=self.constrained_actions)
        if self.constrained_actions:
            act_prompt += "\n\n" + action_contract(self.enabled_tools)
        act_raw = self.action_generate(self.system, [{"role": "user", "content": act_prompt}])
        if self.constrained_actions:
            from redteamrl.harness.protocol import validate_action
            thoughts, action = "", validate_action(act_raw, self.enabled_tools)
        else:
            thoughts, action = parse_action_generation(act_raw)

        # What the planner proposed in its scratchpad (if it embedded a JSON action) vs. what the
        # action step emitted -- the plan/action agreement diagnostic for the two-call design.
        from redteamrl.harness.protocol import _INVALID_ATTACKER_OUTPUT
        _, proposed = parse_action_generation(sp_raw) if sp_raw else ("", None)
        if proposed is not None and proposed.kind == "message" and proposed.text == _INVALID_ATTACKER_OUTPUT:
            proposed = None   # the scratchpad embedded no action; nothing to compare
        self._actions.append(action)
        self.turn_records.append(ScratchpadTurn(sp_text, thoughts, sp_prompt, sp_raw,
                                                act_prompt, act_raw, action, proposed_action=proposed))
        return action
