import json
from redteamrl.sft.filter import is_grounded


def render_target(reasoning: str, action: dict) -> str:
    """Render the attacker's native format: reasoning prose followed by JSON action.

    The PromptedAttacker parses by taking the LAST top-level JSON object,
    so this format is re-parseable by the same path.
    """
    return f"{reasoning}\n{json.dumps(action)}"


def build_example(
    system: str, observable_prompt: str, reasoning: str, action: dict, canary: str
) -> dict | None:
    """Build an SFT example if reasoning is grounded, else drop it.

    A turn whose reasoning cites a canary its observable prompt never contained is dropped.
    Returns {"system", "prompt", "target"} if grounded, else None.
    """
    if not is_grounded(reasoning, [canary], observable_prompt):
        return None

    return {
        "system": system,
        "prompt": observable_prompt,
        "target": render_target(reasoning, action),
    }
