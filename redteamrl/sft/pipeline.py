from redteamrl.sft.attribute import correct_verdict, sensitive_spans
from redteamrl.sft.rationalize import rationalize
from redteamrl.sft.filter import verdict_matches, is_grounded, redaction_effective, redaction_ok
from redteamrl.sft.example import strip_to_sft


def _candidate_example(rec: dict, correct: str, trace: dict | None) -> tuple[dict | None, str]:
	if trace is None:
		return None, "unparseable_rationalization"
	if not verdict_matches(trace, correct):
		return None, "verdict_mismatch"
	if not is_grounded(
		trace["reasoning"], rec["forbidden"], rec["observable_prompt"], rec["required"],
		authoring_values=sensitive_spans(
			rec.get("raw_result") or "",
			rec["forbidden"],
		),
	):
		return None, "ungrounded_reasoning"
	if trace["verdict"] == "redact" and not redaction_effective(
		trace.get("remove") or [], rec.get("raw_result") or "", rec["forbidden"], rec["required"]
	):
		return None, "ineffective_redaction"
	try:
		example = strip_to_sft(rec, trace)
	except ValueError:
		return None, "privileged_prompt"
	example["_class"] = correct
	# Minimality is a reported diagnostic, never an acceptance criterion: the deployment reward
	# scores only whether the secret is gone.
	example["_exact_span"] = (
		redaction_ok(trace.get("remove") or [], rec.get("raw_result") or "",
		             rec["forbidden"], rec["required"])
		if trace["verdict"] == "redact" else None
	)
	return example, "accepted"


# Stages a verdict-conditioned authoring attempt can reach, in order. The model is GIVEN the
# verdict, so this measures rationale authoring, not independent inference -- see blind_review.
AUTHORING_STAGES = (
	"unparseable_rationalization",
	"verdict_mismatch",
	"ungrounded_reasoning",
	"ineffective_redaction",
	"privileged_prompt",
	"accepted",
)


def process_record(rec: dict, generate, max_redact_attempts: int = 1) -> dict:
	"""Author and verify one SFT example.

	Only mechanically labelled ``redact`` records receive best-of-k retries. The first candidate
	that passes every existing verifier is retained, so one observable decision can contribute at
	most one training example.
	"""
	if max_redact_attempts < 1:
		raise ValueError("max_redact_attempts must be at least 1")
	c = correct_verdict(rec)
	example = None
	reason = "ambiguous_label"
	attempt_reasons: list[str] = []
	if c is not None:
		attempt_limit = max_redact_attempts if c == "redact" else 1
		for _ in range(attempt_limit):
			trace = rationalize(rec, generate)
			example, reason = _candidate_example(rec, c, trace)
			attempt_reasons.append(reason)
			if example is not None:
				break
	return {
		"record_id": rec["record_id"],
		"target_verdict": c,
		"accepted": example is not None,
		"reason": reason,
		"attempt_count": len(attempt_reasons),
		"attempt_reasons": attempt_reasons,
		"example": example,
	}
