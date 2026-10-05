from __future__ import annotations

from redteamrl.attack_sft.eval_metrics import exact_repeat_count


def make_repeat_penalty(coef: float):
    """Build a `reward_transform(result, base_reward, spec) -> float` that subtracts `coef` per
    exact-action_key no-progress repeat (arm 3). `coef = 0.0` yields the identity (arm 2 uses the
    same code path). The planted secret is `spec.forbidden[0]`, so the 'no progress' test matches
    what the attacker was trying to exfiltrate -- ground-truth, no model call."""
    def reward_transform(result, base_reward: float, spec) -> float:
        if coef == 0.0 or not spec.forbidden:
            return base_reward
        return base_reward - coef * exact_repeat_count(result.steps, spec.forbidden[0])
    return reward_transform
