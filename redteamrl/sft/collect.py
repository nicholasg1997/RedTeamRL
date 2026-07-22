from redteamrl.harness.episode import run_episode
from redteamrl.policies.prompted import PromptedDefender
from redteamrl.harness.protocol import render_action, render_conversation
from redteamrl.policies.prompted import CALL_INSTRUCTION, RESULT_INSTRUCTION
from redteamrl.sft.attribute import outcome_flags
from redteamrl.types import DefenderDecision


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