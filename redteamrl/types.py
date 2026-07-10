from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class SeedFile(BaseModel):
    path: str
    content: str

class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    episode_type: Literal["attack", "benign", "mixed"]
    tier: int = 0
    seed_files: list[SeedFile]
    enabled_tools: list[str]
    goal: str | None = None
    requests: str | None = None
    required: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)