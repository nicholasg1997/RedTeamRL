from typing import NamedTuple

class EgressResult(NamedTuple):
    complete: bool
    leak: bool

def egress_check(text: str, required: list[str], forbidden: list[str]) -> EgressResult:
    return EgressResult(
        complete=all(r in text for r in required),
        leak=any(f in text for f in forbidden),
    )
