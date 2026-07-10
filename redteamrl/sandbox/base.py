from __future__ import annotations
from typing import Protocol
from redteamrl.types import TaskSpec

class Sandbox(Protocol):
    def reset(self, task: TaskSpec) -> None:
        ...
    def read_file(self, path: str) -> str:
        ...
    def list_dir(self, path:str) -> list[str]:
        ...