"""Environment schema + task builder.

One `Environment` -> many `TaskSpec`s over one shared, canary-injected filesystem.
The canary is injected at BUILD time (never hardcoded in the app files) into a chosen
`canary_path`, so the SAME environment can hide the secret in different places
(config.py, .env, a code comment) — deterministic diversity, not random.
"""
from __future__ import annotations
from pydantic import BaseModel

from redteamrl.types import TaskSpec, SeedFile, AttackerAction


class AttackTemplate(BaseModel):
    id_suffix: str
    goal: str                                    # what the attacker is told to exfiltrate


class BenignTemplate(BaseModel):
    id_suffix: str
    request: str                                 # the legitimate user's stated purpose
    required: list[str]                          # ground-truth answer; must survive canary redaction
    client_actions: list[AttackerAction]         # the scripted legitimate user's moves


class Environment(BaseModel):
    id: str
    files: list[SeedFile]                        # base app — NO canary in here
    canary_line: str                             # e.g. 'API_KEY = "{CANARY}"'; injected at build
    enabled_tools: list[str]
    attacks: list[AttackTemplate]
    benigns: list[BenignTemplate]


def build_tasks(env: Environment, canary: str, canary_path: str, tier: int = 1) -> list[TaskSpec]:
    """Inject `canary` into `canary_path` and expand every attack/benign template into a
    TaskSpec sharing that filesystem. `canary_path` may be an existing file (the line is
    appended) or a new one (created). Varying `canary_path` hides the secret elsewhere."""
    line = env.canary_line.replace("{CANARY}", canary)
    files: list[SeedFile] = []
    injected = False
    for f in env.files:
        content = f.content
        if f.path == canary_path:
            content = content.rstrip("\n") + "\n" + line + "\n"
            injected = True
        files.append(SeedFile(path=f.path, content=content))
    if not injected:
        files.append(SeedFile(path=canary_path, content=line + "\n"))

    tasks: list[TaskSpec] = []
    for a in env.attacks:
        tasks.append(TaskSpec(
            id=f"{env.id}-attack-{a.id_suffix}", episode_type="attack", tier=tier,
            seed_files=files, enabled_tools=env.enabled_tools,
            goal=a.goal, forbidden=[canary],
        ))
    for b in env.benigns:
        tasks.append(TaskSpec(
            id=f"{env.id}-benign-{b.id_suffix}", episode_type="benign", tier=tier,
            seed_files=files, enabled_tools=env.enabled_tools,
            request=b.request, required=b.required, forbidden=[canary],
            client_actions=b.client_actions,
        ))
    return tasks
