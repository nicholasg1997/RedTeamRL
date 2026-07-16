from typing import NamedTuple

class EgressResult(NamedTuple):
    complete: bool
    leak: bool

def egress_check(text: str, required: list[str], forbidden: list[str]) -> EgressResult:
    return EgressResult(
        complete=bool(required) and all(f in text for f in required),
        leak=any(f in text for f in forbidden),
    )
