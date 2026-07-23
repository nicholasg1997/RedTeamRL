from redteamrl.sft.attribute import correct_verdict
from redteamrl.sft.rationalize import rationalize
from redteamrl.sft.filter import verdict_matches, is_grounded, redaction_ok
from redteamrl.sft.example import strip_to_sft

def process_record(rec: dict, generate) -> dict:
	c = correct_verdict(rec)
	example = None
	reason = 'ambiguous_label'
	trace = rationalize(rec, generate) if c is not None else None
	if c is not None and trace is None:
		reason = "unparseable_rationalization"
	elif trace is not None and not verdict_matches(trace, c):
		reason = "verdict_mismatch"
	elif trace is not None and not is_grounded(
			trace["reasoning"], rec["forbidden"], rec["observable_prompt"], rec["required"]
	):
		reason = "ungrounded_reasoning"
	elif trace is not None and trace["verdict"] == "redact" and not redaction_ok(
			trace.get("remove") or [], rec.get("raw_result") or "", rec["forbidden"], rec["required"]
	):
		reason = "invalid_redaction"
	elif trace is not None:
		try:
			example = strip_to_sft(rec, trace)
			example["_class"] = c
			reason = "accepted"
		except ValueError:
			reason = "privileged_prompt"
	return {"record_id": rec["record_id"], "accepted": example is not None,
	        "reason": reason, "example": example}
