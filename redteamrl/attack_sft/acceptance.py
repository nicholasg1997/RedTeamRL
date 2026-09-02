def accept(before: dict, after: dict, rank_floor: float = 0.9) -> tuple[bool, list[str]]:
    """Promotion gate: held-out win-rate + entropy-collapse veto.

    Promote (True, []) iff after["win_rate"] > before["win_rate"]
    AND after["effective_rank"] >= rank_floor * before["effective_rank"].

    Args:
        before: dict with "win_rate" and "effective_rank" from baseline
        after: dict with "win_rate" and "effective_rank" from SFT
        rank_floor: minimum rank retention ratio (default 0.9 = 90%)

    Returns:
        (ok, reasons) where ok is True if promotion succeeds, and reasons
        is a list of strings naming each failed condition. Returns (True, [])
        if all conditions pass.
    """
    reasons = []

    # Check win_rate improvement
    if after["win_rate"] <= before["win_rate"]:
        reasons.append(f"win_rate not improved: {before['win_rate']} -> {after['win_rate']}")

    # Check effective_rank preservation
    min_rank = rank_floor * before["effective_rank"]
    if after["effective_rank"] < min_rank:
        reasons.append(f"rank collapsed: {before['effective_rank']} -> {after['effective_rank']} (floor: {min_rank})")

    return (len(reasons) == 0, reasons)
