def accept(before: dict, after: dict, rank_floor: float = 0.9,
           win_epsilon: float = 0.05, max_no_progress_rate: float = 0.25) -> tuple[bool, list[str]]:
    """Promotion gate for the reasoning-first SFT (spec 2026-09-03 §3.3).

    SFT's job is a NON-COLLAPSING bootstrap, not a win-rate climb (that is GRPO's). Promote iff:
      1. no win collapse: after.win_rate >= before.win_rate - win_epsilon
      2. reasoning entropy held: after.effective_rank >= rank_floor * before.effective_rank
      3. action non-collapse: after.repeated_no_progress_rate <= max_no_progress_rate

    Condition 3 is load-bearing: in the failed run eff_rank HELD while the action distribution
    collapsed, because eff_rank probes reasoning representations, not actions.
    """
    reasons = []
    if after["win_rate"] < before["win_rate"] - win_epsilon:
        reasons.append(
            f"win collapsed: {before['win_rate']} -> {after['win_rate']} "
            f"(floor {before['win_rate'] - win_epsilon})")
    min_rank = rank_floor * before["effective_rank"]
    if after["effective_rank"] < min_rank:
        reasons.append(
            f"rank collapsed: {before['effective_rank']} -> {after['effective_rank']} (floor {min_rank})")
    repeat = after.get("repeated_no_progress_rate", 0.0)
    if repeat > max_no_progress_rate:
        reasons.append(
            f"action collapse: repeated_no_progress_rate {repeat:.2f} > {max_no_progress_rate}")
    return (len(reasons) == 0, reasons)
