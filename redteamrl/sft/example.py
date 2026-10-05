import json

from redteamrl.sft.filter import _PRIVILEGED_PHRASES

TARGET_RENDER_VERSION = 2


def render_target(trace: dict) -> str:
	decision = {"reasoning": trace["reasoning"], "verdict": trace["verdict"]}
	if trace.get("remove") is not None:
		decision["remove"] = trace["remove"]
	if trace.get("content") is not None:
		decision["content"] = trace["content"]
	# Keep reasoning before the verdict inside a single parseable object. The previous format
	# supervised the same rationale twice (plain prose, then again in JSON), so most of the token
	# loss optimized prose imitation rather than the security decision.
	return json.dumps(decision)

def strip_to_sft(record: dict, trace:dict) -> dict:
	prompt = record["observable_prompt"]
	low = prompt.lower()
	if any(p in low for p in _PRIVILEGED_PHRASES):
		raise ValueError("observable prompt contains a privileged phrase — hindsight leaked into input")
	return {"system": record["system"], "prompt": prompt, "target": render_target(trace)}
