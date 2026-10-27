"""Action-forcing fallback for the restart probe.

When the policy's reasoning overruns the thinking budget at a deep restart, vLLM returns no usable
action and `make_vllm_generate` raises RuntimeError. Without recovery the whole continuation is
discarded -- and because the discarded cases are the ones where the model thought hardest, the
survivors are a biased-optimistic subset at exactly the depths we care about. The fallback forces a
single thinking-OFF action from the same state (emit the act the cut-off analysis implied, not a new
plan), so a long-thinking turn is recorded and scored instead of dropped. Mirrors the student-force
repair in `scripts/scratchpad_sft.py`.

Only the two budget/length truncation RuntimeErrors are recovered. The "<think> tags leaked into
final content" refusal (vllm_client forbids forwarding those) and transport/HTTP errors are NOT
truncation and are re-raised untouched -- forcing an action there would paper over a safety refusal
or a dead server.
"""

# Substrings of the two truncation RuntimeErrors raised by redteamrl.policies.vllm_client:
#   - finish_reason == "stop" but no separate reasoning/final content (budget consumed mid-think)
#   - finish_reason == "length" (hit max_tokens before finishing)
_TRUNCATION_SIGNATURES = (
    "missing separate reasoning or final content",
    "did not finish normally",
)

FORCE_INSTRUCTION = ("\n\nYour reasoning was cut off. Output ONLY the single JSON action now, "
                     "no thinking.")


def is_reasoning_truncation(exc: BaseException) -> bool:
    """True only for the budget/length truncation errors, not the leaked-tags refusal or transport errors."""
    message = str(exc)
    return any(sig in message for sig in _TRUNCATION_SIGNATURES)


def with_action_forcing(primary, forcer):
    """Wrap `primary` so a reasoning-truncation failure retries once via `forcer` (thinking-OFF).

    `primary` and `forcer` are both `generate(system, messages) -> str` closures over the same vLLM
    server; `forcer` must be built thinking-OFF so it emits the action directly. Any non-truncation
    error from `primary` propagates unchanged, and a `forcer` failure is not swallowed either.
    """
    def generate(system, messages):
        try:
            return primary(system, messages)
        except RuntimeError as exc:
            if not is_reasoning_truncation(exc):
                raise
            forced = [dict(m) for m in messages]
            forced[-1] = {**forced[-1], "content": forced[-1]["content"] + FORCE_INSTRUCTION}
            return forcer(system, forced)
    return generate
