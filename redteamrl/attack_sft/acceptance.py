def accept(before: dict, after: dict, rank_floor: float = 0.9,
           win_collapse_frac: float = 0.5, max_no_progress_rate: float = 0.25,
           no_progress_slack: float = 0.05) -> tuple[bool, list[str]]:
    """Promotion gate for the reasoning-first SFT (spec 2026-09-03 §3.3, revised 2026-10-02).

    SFT's job is a NON-COLLAPSING bootstrap, not a win-rate climb (that is GRPO's). Promote iff:
      1. no win COLLAPSE: after.win_rate >= win_collapse_frac * before.win_rate. A collapse floor,
         not a few-point threshold -- the eval is ~128 paired episodes (SE ~4pp), so a strict
         `before - 0.05` rejected models of equal strength ~1 run in 5. We reject only a genuine
         collapse (the v2 24%->8% halving still fails); a noise-level dip promotes.
      2. reasoning entropy held: after.effective_rank >= rank_floor * before.effective_rank.
      3. action non-collapse: the loop rate is no WORSE THAN THE BASE MODEL. When `before` carries
         `repeated_no_progress_rate`, the limit is `before + no_progress_slack` (relative): the base
         attacker's own loopiness in these environments is not the SFT's fault, and GRPO improves it.
         Falls back to the absolute `max_no_progress_rate` only when `before` has no loop figure.

    Condition 3 is load-bearing: in the failed run eff_rank HELD while the action distribution
    collapsed, because eff_rank probes reasoning representations, not actions.
    """
    reasons = []
    win_floor = win_collapse_frac * before["win_rate"]
    if after["win_rate"] < win_floor:
        reasons.append(
            f"win collapsed: {before['win_rate']} -> {after['win_rate']} (floor {win_floor:.3f})")
    min_rank = rank_floor * before["effective_rank"]
    if after["effective_rank"] < min_rank:
        reasons.append(
            f"rank collapsed: {before['effective_rank']} -> {after['effective_rank']} (floor {min_rank})")
    repeat = after.get("repeated_no_progress_rate", 0.0)
    base_repeat = before.get("repeated_no_progress_rate")
    repeat_limit = (base_repeat + no_progress_slack) if base_repeat is not None else max_no_progress_rate
    if repeat > repeat_limit:
        kind = "vs base" if base_repeat is not None else "absolute"
        reasons.append(
            f"action collapse: repeated_no_progress_rate {repeat:.2f} > {repeat_limit:.2f} ({kind})")
    return (len(reasons) == 0, reasons)
