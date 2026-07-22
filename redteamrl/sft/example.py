import json

from redteamrl.sft.filter import _PRIVILEGED_PHRASES

def render_target(trace: dict) -> str:
	decision = {"reasoning": trace["reasoning"], "verdict": trace["verdict"]}
	if trace.get("remove") is not None:
		decision["remove"] = trace["remove"]
	if trace.get("content") is not None:
		decision["content"] = trace["content"]
	return trace["reasoning"] + "\n" + json.dumps(decision)

def strip_to_sft(record: dict, trace:dict) -> dict:
	prompt = record["observable_prompt"]
	low = prompt.lower()
	if any(p in low for p in _PRIVILEGED_PHRASES):
		raise ValueError("observable prompt contains a privileged phrase — hindsight leaked into input")
	return {"system": record["system"], "prompt": prompt, "target": render_target(trace)}