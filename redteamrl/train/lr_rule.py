"""KL-adaptive learning rate for attacker GRPO (lineage v7).

How far one iteration's update moves the policy -- the behaviour-to-new KL, `kl_iter` -- is what
actually matters, and Adam's per-step movement is ~LR, so a fixed LR is a guess about a quantity we
can measure. This rule turns the measurement into the next iteration's LR: too little movement
raises it, too much (or a target-KL brake) lowers it, inside a hold band. It is deterministic given
the checkpointed LR, so a resumed run follows the identical schedule, and a lineage never needs a
mid-run LR change.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LrRule:
    initial: float = 3e-5
    target_kl: float = 0.01     # per-iteration behaviour->new KL (mean per token)
    factor: float = 1.5
    lo: float = 1e-5
    hi: float = 3e-4
    version: str = "kl-adaptive-v1"

    def as_identity(self) -> dict:
        return {f"lr_{k}": v for k, v in self.__dict__.items()}


def next_lr(lr: float, kl_iter: float, early_stopped: bool, rule: LrRule,
            steps: int | None = None) -> float:
    """Hold inside [target/2, 2*target]; step by `factor` outside it; clamp to [lo, hi].

    `steps == 0` (an all-dead iteration, or a brake on the very first minibatch) holds: no step was
    taken, so `kl_iter` measured nothing and must not move the LR."""
    if steps == 0:
        return lr
    if early_stopped or kl_iter > 2 * rule.target_kl:
        lr = lr / rule.factor
    elif kl_iter < rule.target_kl / 2:
        lr = lr * rule.factor
    return min(rule.hi, max(rule.lo, lr))
