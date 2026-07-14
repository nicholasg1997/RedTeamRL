from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Literal

class SeedFile(BaseModel):
    path: str
    content: str

class AttackerAction(BaseModel):
    kind: Literal["message", "tool_call"]
    text: str | None = None
    tool: str | None = None
    args: dict[str, str] | None = None

    @model_validator(mode="after")
    def _fields_match_kind(self) -> "AttackerAction":
        if self.kind == "message" and self.text is None:
            raise ValueError("message action requires `text`")
        if self.kind == "tool_call" and (self.tool is None or self.args is None):
            raise ValueError("tool_call action requires `tool` and `args`")
        return self

class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    episode_type: Literal["attack", "benign", "mixed"]
    tier: int = 0
    seed_files: list[SeedFile]
    enabled_tools: list[str]
    goal: str | None = None
    request: str | None = None
    required: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    client_actions: list[AttackerAction] | None = None

    @model_validator(mode="after")
    def _fields_match_ep_type(self) -> "TaskSpec":
        if self.episode_type == "attack":
            if self.goal is None:
                raise ValueError("attack task must have a `goal`")
            if not self.forbidden:
                raise ValueError("attack task must have a non-empty `forbidden`")
            if self.client_actions is not None:
                raise ValueError("attack task must not have `client_actions` (the LLM attacker drives)")

        if self.episode_type == "benign":
            if self.request is None:
                raise ValueError("benign task must have a `request`")
            if not self.required:
                raise ValueError("benign task must have a non-empty `required`")
            if not self.client_actions:
                raise ValueError("benign task must have `client_actions` (the LLM defender drives)")

        # TODO: add check for mixed tasks

        return self

class DefenderDecision(BaseModel):
    reasoning: str
    verdict: Literal["allow", "reject", "pass", "redact", "respond"]
    content: str | None = None