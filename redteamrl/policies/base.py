from __future__ import annotations
from typing import Protocol, runtime_checkable
from redteamrl.types import AttackerAction, DefenderDecision

# The episode harness interacts with the two roles ONLY through these shapes.
# Each policy owns its own wire format (prompting + parsing); the harness deals
# in typed AttackerAction / DefenderDecision objects, never raw model text.

Conversation = list[dict]   # ordered turn records; each policy renders its own view

@runtime_checkable
class AttackerPolicy(Protocol):
    def act(self, conversation: Conversation) -> AttackerAction:
        """Produce the attacker's next move (already parsed)."""
        ...

@runtime_checkable
class DefenderPolicy(Protocol):
    def adjudicate_call(self, conversation: Conversation, action: AttackerAction) -> DefenderDecision:
        """Decide allow/reject for a proposed tool call."""
        ...

    def adjudicate_result(self, conversation: Conversation, action: AttackerAction, raw_result: str) -> DefenderDecision:
        """Decide pass/redact for a tool result before it reaches the attacker."""
        ...

    def respond(self, conversation: Conversation, action: AttackerAction) -> DefenderDecision:
        """Respond to a natural-language message from the attacker."""
        ...