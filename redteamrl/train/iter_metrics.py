"""Build the per-iteration metric dicts the dashboard reads (see train/metrics_log.py).

Keys are `<category>.<name>`; the prefix is the dashboard section. Both builders share `_common`,
so learning/diversity/throughput/update keys cannot drift between the attacker and defender phases.
Pure functions over dicts the drivers already compute: stdlib only, no torch.
"""
from __future__ import annotations


def _common(*, live, n_examples, div, rew, tactics, update, rollout_min, update_min, gen_tokens,
            peak_gib) -> dict:
    out = {
        "learning.live_frac": live / n_examples if n_examples else None,
        "learning.dead_groups": div["dead_group_rate"],
        "learning.mixed_reward": div["mixed_reward_group_rate"],
        "learning.mean_reward": rew["mean_reward"],
        "diversity.encoded_pass_rate": tactics["encoded_pass_rate"],
        "throughput.rollout_min": rollout_min,
        "throughput.update_min": update_min,
        "throughput.gen_tokens": gen_tokens,
        "throughput.tokens_per_s": gen_tokens / (rollout_min * 60) if rollout_min else None,
        "throughput.peak_gib": peak_gib,
        "update.loss": update["loss"],
        "update.kl": update["mean_kl"],
        "update.grad_norm": update["grad_norm"],
        "update.ratio": update["mean_ratio"],
    }
    # v7 minibatch-update keys: present only when the update reported them (absent, never 0).
    if "steps" in update:
        out.update({
            "update.steps": update["steps"],
            "update.early_stopped": int(update["early_stopped"]),
            "update.kl_iter": update["kl_iter"],
            "update.clip_frac": update["clip_frac"],
        })
    if "lr" in update:
        out["update.lr"] = update["lr"]
    for tactic, count in tactics["leaks"].items():
        out[f"diversity.leaks.{tactic}"] = count
    return out


def attacker_metrics(*, live: int, n_examples: int, div: dict, rew: dict, summary: dict,
                     tactics: dict, update: dict, rollout_min: float, update_min: float,
                     gen_tokens: int, peak_gib: float, banks_size: int, win_routes: int,
                     heldout: dict | None = None) -> dict:
    out = _common(live=live, n_examples=n_examples, div=div, rew=rew, tactics=tactics,
                  update=update, rollout_min=rollout_min, update_min=update_min,
                  gen_tokens=gen_tokens, peak_gib=peak_gib)
    out.update({
        "learning.fresh_win": summary["fresh_win_rate"],
        "learning.fresh_win_task_avg": summary["fresh_win_rate_task_avg"],
        "restart.win_rate": summary["restart_win_rate"],
        "restart.fallbacks": summary["fallbacks"],
        "restart.groups": summary["n_restart_groups"],
        "diversity.win_routes": win_routes,
        "diversity.bank_size": banks_size,
        "diversity.banked_wins": summary["banked_wins"],
        "diversity.banked_near": summary["banked_near"],
    })
    # v6 dynamic-sampling keys exist only when the summary came from finalize_v6 (absent for v5).
    if "dead_before" in summary:
        n_base = summary["n_base_groups"]
        out.update({
            "restart.dead_before": summary["dead_before"] / n_base if n_base else None,
            "restart.dead_after": summary["dead_after"] / n_base if n_base else None,
            "restart.replacements": summary["replacements"],
            "restart.replacements_live": summary["replacements_live"],
            "restart.depth_frac_mean": summary["depth_frac_mean"],
            "restart.backed_off_tasks": len(summary["backed_off"]),
        })
        for bucket, rate in summary["win_by_depth"].items():
            out[f"restart.win_by_depth.{bucket}"] = rate
    # v7 graduation / same-iteration banking keys exist only when the summary came from finalize_v7.
    if "graduated" in summary:
        out.update({
            "restart.graduated": summary["graduated"],
            "restart.graduated_total": summary["graduated_total"],
            "restart.same_iter_replacements": summary["same_iter_replacements"],
            "restart.same_iter_replacements_live": summary["same_iter_replacements_live"],
        })
    # Held-out keys exist only on iterations that ran the eval: ABSENT, never 0, so the chart
    # shows sparse points instead of a fake drop to zero.
    if heldout is not None:
        out.update({f"heldout.{k}": heldout[k] for k in ("win_rate", "stalled_rate", "dead_rate")})
    return out


def defender_metrics(*, live: int, n_examples: int, div: dict, rew: dict, decomp: dict,
                     tactics: dict, update: dict, rollout_min: float, update_min: float,
                     gen_tokens: int, peak_gib: float) -> dict:
    out = _common(live=live, n_examples=n_examples, div=div, rew=rew, tactics=tactics,
                  update=update, rollout_min=rollout_min, update_min=update_min,
                  gen_tokens=gen_tokens, peak_gib=peak_gib)
    out.update({
        "defender.attack_leak": decomp["attack_policy_leak"],
        "defender.benign_leak": decomp["benign_policy_leak"],
        "defender.protocol_fail": decomp["protocol_failure"],
        "defender.over_refusal": decomp["refuse"],
        "defender.clean": decomp["clean"],
    })
    return out
