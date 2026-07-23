import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from redteamrl.harness.episode import run_episode
from redteamrl.policies.prompted import PromptedDefender
from redteamrl.harness.protocol import render_action, render_conversation
from redteamrl.policies.prompted import CALL_INSTRUCTION, RESULT_INSTRUCTION
from redteamrl.sft.attribute import outcome_flags
from redteamrl.types import DefenderDecision

def _write_json_atomic(path, obj):
	tmp = path + ".tmp"
	with open(tmp, "w") as f:
		json.dump(obj, f)
	os.replace(tmp, path)

def _run_one(spec, rollout_index, agent_for, defender_factory, sandbox_factory, out_dir, max_turns):
	key = f"{spec.id}#{rollout_index}"
	path = os.path.join(out_dir, f"{key}.json")
	if os.path.exists(path):
		return False
	defender = defender_factory()
	sandbox = sandbox_factory()
	try:
		ep = run_episode(spec, agent_for(spec), defender, sandbox, max_turns=max_turns)
	finally:
		sandbox.close()
	leaked, delivered = outcome_flags(ep)
	records = []
	for decision_index, rec in enumerate(defender.records):
		rec.update({"episode_key": key, "task_id": spec.id, "episode_type": spec.episode_type,
		            "record_id": f"{key}:decision-{decision_index}",
		            "system": defender.system, "forbidden": spec.forbidden, "required": spec.required,
		            "true_role": spec.episode_type,
		            "episode_leaked": leaked, "episode_required_delivered": delivered})
		records.append(rec)
	_write_json_atomic(path, {"episode_key": key, "records": records})
	return True

def collect_episodes(tasks, n_collect, agent_for, defender_factory, sandbox_factory,
					 out_dir, max_turns, max_workers=1, commit=None, commit_every=None):
	"""Collect independent episodes concurrently and persist completed waves.

	Only newly written episodes count toward commits. Work is submitted one worker-sized wave at a
	time, so a commit never races with another episode file being written in this process.
	"""
	os.makedirs(out_dir, exist_ok=True)
	all_work = [(spec, r) for spec in tasks for r in range(n_collect)]
	work = [
		(spec, r) for spec, r in all_work
		if not os.path.exists(os.path.join(out_dir, f"{spec.id}#{r}.json"))
	]
	skipped = len(all_work) - len(work)
	written = 0
	since_commit = 0
	commit_every = max_workers if commit_every is None else commit_every
	if commit_every <= 0:
		raise ValueError("commit_every must be positive")

	with ThreadPoolExecutor(max_workers=max_workers) as pool:
		for start in range(0, len(work), max_workers):
			wave = work[start:start + max_workers]
			futures = [pool.submit(_run_one, spec, r, agent_for, defender_factory,
			                       sandbox_factory, out_dir, max_turns) for spec, r in wave]
			for fut in as_completed(futures):
				if fut.result():
					written += 1
					since_commit += 1
				else:
					skipped += 1
			if commit and since_commit >= commit_every:
				commit()
				since_commit = 0

	if commit and since_commit:
		commit()
	return {"expected": len(all_work), "skipped": skipped, "written": written}


class RecordingDefender(PromptedDefender):

	def __init__(self, generate, max_context_chars=None):
		super().__init__(generate, max_context_chars)
		self.records: list[dict] = []

	def adjudicate_call(self, conversation, action) -> DefenderDecision:
		prompt = render_conversation(conversation, max_chars=self.max_context_chars) + "\n\n" + \
            CALL_INSTRUCTION.replace("{action}", render_action(action))
		decision = super().adjudicate_call(conversation, action)
		self.records.append({"decision_type": "call", "observable_prompt": prompt,
                             "raw_result": None, "decision": decision.model_dump()})
		return decision

	def adjudicate_result(self, conversation, action, raw_result) -> DefenderDecision:
		prompt = render_conversation(conversation, max_chars=self.max_context_chars) + "\n\n" + \
            RESULT_INSTRUCTION.replace("{action}", render_action(action)).replace("{raw_result}", raw_result)
		decision = super().adjudicate_result(conversation, action, raw_result)
		self.records.append({"decision_type": "result", "observable_prompt": prompt,
		                     "raw_result": raw_result, "decision": decision.model_dump()})
		return decision

def collect_records(tasks, attack_factory, benign_factory, defender_factory, sandbox_factory, n_rollouts, max_turns):
	out = []
	for spec in tasks:
		factory = attack_factory if spec.episode_type == "attack" else benign_factory
		for _ in range(n_rollouts):
			defender = defender_factory()
			sandbox = sandbox_factory()
			try:
				ep = run_episode(spec, factory(spec), defender, sandbox, max_turns=max_turns)
			finally:
				sandbox.close()
			leaked, delivered = outcome_flags(ep)
			for r in defender.records:
				r.update({"task_id": spec.id, "episode_type": spec.episode_type,
				          "system": defender.system, "forbidden": spec.forbidden,
				          "required": spec.required, "true_role": spec.episode_type,
				          "episode_leaked": leaked, "episode_required_delivered": delivered})
				out.append(r)
	return out
