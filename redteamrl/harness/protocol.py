import json
from redteamrl.types import AttackerAction

def parse_attacker_action(raw: str) -> AttackerAction:
    obj = _extract_json(raw)
    if isinstance(obj, dict):
        try:
            return AttackerAction(**obj)
        except Exception:
            pass
    return AttackerAction(kind="message", text=raw)

def _extract_json(raw: str) -> dict | None:
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = json.JSONDecoder().raw_decode(raw[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None