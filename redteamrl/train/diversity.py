"""Within-group variance diagnostics for GRPO.

An exposed weakness is not automatically a learning signal. GRPO's baseline is the group mean, so
if every sample in a group produces the same reward the advantage is zero and the group teaches
nothing -- however wrong that shared behaviour is. A defender that allows every reconstruction call
has enormous headroom and zero gradient.

These are measured at the GRPO SAMPLING temperature, not the eval's temp=0. At temp=0 a policy is
constant by construction, so the eval cannot see this at all. The response to a dead group is more
exploration (temperature, group size), never answer-key prompt engineering -- that would close the
attacker's foothold instead of teaching the defender to close it.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    raw = -sum((n / total) * math.log2(n / total) for n in counts.values() if n)
    # Normalize by the maximum achievable given the number of distinct verdicts observed, so a
    # clean 50/50 split reads as 1.0 regardless of how many verdicts the protocol allows.
    return raw / math.log2(len(counts))


def _field(example: Any, name: str):
    return example[name] if isinstance(example, dict) else getattr(example, name)


def _typed_verdict(entry: Any) -> tuple[str, str] | None:
    """Normalize current typed verdicts and legacy string-only diagnostics."""
    if isinstance(entry, dict):
        decision_type = entry.get("decision_type")
        verdict = entry.get("verdict")
        if decision_type and verdict:
            return str(decision_type), str(verdict)
        return None
    verdict = str(entry)
    if verdict in {"allow", "reject"}:
        return "call", verdict
    if verdict in {"pass", "redact"}:
        return "result", verdict
    if verdict == "respond":
        return "response", verdict
    return "unknown", verdict


def reward_summary(examples: list[Any]) -> dict:
    """Mean episode reward, overall and per track — the scalar the policy is actually optimizing.

    `loss` is ~0 by construction in GRPO (group-normalized advantages sum to zero), so it says
    nothing about progress. The component rates (attack leak, benign leak) move independently and
    can trade against each other, which makes "is this improving?" hard to read from them alone.
    Mean reward is the single number that separates a policy climbing from one cycling between
    failure modes: an oscillating policy shows anti-correlated components with a FLAT mean, a
    learning one shows a rising mean even while the components wobble.

    Deduplicated by episode: reward is per episode but examples are per decision, so averaging raw
    examples would weight long trajectories by their length.
    """
    by_episode: dict[Any, tuple[str, float]] = {}
    for example in examples:
        episode_id = _field(example, "episode_id")
        if episode_id not in by_episode:
            by_episode[episode_id] = (
                str(_field(example, "episode_type")), float(_field(example, "reward"))
            )

    rewards = [reward for _, reward in by_episode.values()]
    attack = [reward for kind, reward in by_episode.values() if kind == "attack"]
    benign = [reward for kind, reward in by_episode.values() if kind == "benign"]

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "n_episodes": len(by_episode),
        "mean_reward": mean(rewards),
        "mean_reward_attack": mean(attack),
        "mean_reward_benign": mean(benign),
    }


def group_diversity(examples: list[Any]) -> dict:
    """Aggregate per-task groups into diversity and liveness rates.

    Accepts dicts or Example objects carrying ``task_id``, ``episode_id``, ``reward`` and a
    ``verdicts`` sequence. A group is DEAD when its rewards are all equal — that, not verdict
    variation, is what zeroes the advantage.
    """
    if not examples:
        return {
            "n_groups": 0, "mixed_verdict_group_rate": 0.0, "mixed_reward_group_rate": 0.0,
            "verdict_entropy": 0.0, "dead_group_rate": 0.0, "verdict_counts": {},
            "verdict_counts_by_type": {},
            "mixed_verdict_group_rate_by_type": {},
            "verdict_entropy_by_type": {},
        }

    # A generation capture creates one Example per model call, and every Example from an episode
    # carries the same episode-level reward and verdict trace. Collapse those duplicates before
    # counting or long episodes would be weighted once per captured call.
    episodes_by_group: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for example in examples:
        task_id = str(_field(example, "task_id"))
        episode_id = _field(example, "episode_id")
        if episode_id not in episodes_by_group[task_id]:
            episodes_by_group[task_id][episode_id] = {
                "reward": _field(example, "reward"),
                "verdicts": list(_field(example, "verdicts") or []),
            }

    verdicts_by_group_and_type: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    overall_by_type: dict[str, Counter] = defaultdict(Counter)
    for task_id, episodes in episodes_by_group.items():
        for episode in episodes.values():
            for entry in episode["verdicts"]:
                typed = _typed_verdict(entry)
                if typed is None:
                    continue
                decision_type, verdict = typed
                verdicts_by_group_and_type[task_id][decision_type][verdict] += 1
                overall_by_type[decision_type][verdict] += 1

    mixed_verdict = 0
    mixed_reward = 0
    entropy_total = 0.0
    all_types = sorted(overall_by_type)
    mixed_by_type = Counter()
    entropy_by_type = Counter()
    for task_id, episodes in episodes_by_group.items():
        counters = verdicts_by_group_and_type[task_id]
        observed_entropies = []
        group_mixed = False
        for decision_type in all_types:
            counts = counters.get(decision_type, Counter())
            is_mixed = len(counts) > 1
            entropy = _entropy(counts)
            mixed_by_type[decision_type] += is_mixed
            entropy_by_type[decision_type] += entropy
            if counts:
                observed_entropies.append(entropy)
            group_mixed = group_mixed or is_mixed
        mixed_verdict += group_mixed
        mixed_reward += len({episode["reward"] for episode in episodes.values()}) > 1
        entropy_total += (
            sum(observed_entropies) / len(observed_entropies)
            if observed_entropies else 0.0
        )

    n_groups = len(episodes_by_group)
    flattened_counts = {
        verdict: count
        for counts in overall_by_type.values()
        for verdict, count in counts.items()
    }
    return {
        "n_groups": n_groups,
        "mixed_verdict_group_rate": mixed_verdict / n_groups,
        "mixed_reward_group_rate": mixed_reward / n_groups,
        "verdict_entropy": entropy_total / n_groups,
        # The one that gates: no reward spread means no advantage means no update.
        "dead_group_rate": 1.0 - (mixed_reward / n_groups),
        "verdict_counts": flattened_counts,
        "verdict_counts_by_type": {
            decision_type: dict(counts)
            for decision_type, counts in overall_by_type.items()
        },
        "mixed_verdict_group_rate_by_type": {
            decision_type: mixed_by_type[decision_type] / n_groups
            for decision_type in all_types
        },
        "verdict_entropy_by_type": {
            decision_type: entropy_by_type[decision_type] / n_groups
            for decision_type in all_types
        },
    }
