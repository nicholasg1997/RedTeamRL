import json
import re
from redteamrl.sft.filter import is_grounded


def render_target(reasoning: str, action: dict) -> str:
    """Render the attacker's native format: reasoning prose followed by JSON action.

    The PromptedAttacker parses by taking the LAST top-level JSON object,
    so this format is re-parseable by the same path.
    """
    return f"{reasoning}\n{json.dumps(action)}"


def scrub_secret(text: str, canary: str) -> str:
    """Replace every occurrence of canary with <the secret>.

    Scrubs the key from reasoning so SFT tunes strategy, not key-values.
    """
    return text.replace(canary, "<the secret>")


def _extract_potential_secrets(text: str) -> list[str]:
    """Extract potential secrets from text (e.g., 'sk-...', 'pw-...', 'api-...' patterns)."""
    # Match strings that look like secrets: word-word-word... pattern with dashes
    # Examples: sk-CANARY-7a1b2c, pw-OTHER-xyz, api-key-value
    pattern = r'\b[a-z]+(?:-[a-z0-9]+)+\b'
    return re.findall(pattern, text, re.IGNORECASE)


def build_example(
    system: str, observable_prompt: str, reasoning: str, action: dict, canary: str,
    train_action: bool = True
) -> dict | None:
    """Build an SFT example if reasoning is grounded, else drop it.

    Checks grounding on the ORIGINAL reasoning first (catches the canary and other secrets),
    then scrubs the canary from the reasoning before rendering the target. The (unscrubbed)
    action is kept as-is, so submit actions preserve the real canary value.

    Returns {"system", "prompt", "target", "action_loss_weight"} if grounded, else None.
    action_loss_weight is 1.0 if train_action else 0.0 (for masked actions).
    """
    # Extract all potential secrets from reasoning and check grounding
    potential_secrets = _extract_potential_secrets(reasoning)
    all_secrets_to_check = [canary] + potential_secrets

    # Check grounding on original reasoning with canary and any detected potential secrets
    if not is_grounded(reasoning, all_secrets_to_check, observable_prompt):
        return None

    # Scrub the canary from reasoning for the SFT example
    scrubbed_reasoning = scrub_secret(reasoning, canary)

    return {
        "system": system,
        "prompt": observable_prompt,
        "target": render_target(scrubbed_reasoning, action),
        "action_loss_weight": 1.0 if train_action else 0.0,
    }
