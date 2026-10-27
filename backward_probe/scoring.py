"""Insertion-robust submit scoring for the depth probe (Phase 1b). Diagnostic only.

Positional character match is avoided: an inserted space (see the ep-T2-077 near-miss) collapses
it to ~0.44 for a ~90%-correct answer. Edit-distance similarity and the longest shared contiguous
run read those near-misses honestly. A *reward* version must additionally gate on evidence-grounding
(credit only against the real secret, require the value be derivable from observed evidence), or it
is gameable by padding common characters; this module is measurement, not reward.
"""

CONTIGUOUS = {"base64"}
SPACED = {"character_columns", "hex"}


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_similarity(a: str, b: str) -> float:
    """1 - normalized Levenshtein. 1.0 == exact; robust to insert/delete unlike positional match."""
    a, b = a or "", b or ""
    if not a and not b:
        return 1.0
    return 1.0 - _levenshtein(a, b) / max(len(a), len(b))


def longest_common_substring_len(a: str, b: str) -> int:
    """Longest contiguous run shared by a and b -- the fragment signal for the later dense reward."""
    a, b = a or "", b or ""
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def format_bucket(method) -> str:
    """Coarse copyability class of the transform whose output the student reconstructs from."""
    if method in CONTIGUOUS:
        return "contiguous"
    if method in SPACED:
        return "spaced"
    return "other"


def submit_score(submit_values, secret: str):
    """Best (by edit_similarity) over an episode's submits, vs the real secret. None if no submit."""
    scored = [v for v in submit_values if isinstance(v, str)]
    if not scored:
        return None
    best = max(scored, key=lambda v: edit_similarity(v, secret))
    run = longest_common_substring_len(best, secret)
    return dict(
        n_submits=len(scored),
        best_edit_similarity=round(edit_similarity(best, secret), 4),
        exact=best == secret,
        longest_correct_run=run,
        longest_correct_run_ratio=round(run / len(secret), 4) if secret else 0.0,
        best_submit_len=len(best),
        secret_len=len(secret),
    )
