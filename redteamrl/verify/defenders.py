from redteamrl.types import DefenderDecision, AttackerAction

Conversation = list[dict] 

class NullDefender:
	def adjudicate_call(self, conversation: Conversation, action: AttackerAction) -> DefenderDecision:
		return DefenderDecision(reasoning='[null] allow', verdict='allow',content=None)
	def adjudicate_result(self, conversation: Conversation, action: AttackerAction, raw_result: str) -> DefenderDecision:
		return DefenderDecision(reasoning='[null] pass', verdict='pass',content=None)
	def respond(self, conversation: Conversation, action: AttackerAction) -> DefenderDecision:
		return DefenderDecision(reasoning='[null] respond', verdict='respond',content='(ok)')

class OracleRedactorDefender:
	def __init__(self, forbidden: list[str]):
		self.forbidden = forbidden

	def _scrub(self, text: str) -> str:
		for f in self.forbidden:
			text = text.replace(f, "[REDACTED]")
		return text

	def adjudicate_call(self, conversation: Conversation, action: AttackerAction) -> DefenderDecision:
		return DefenderDecision(reasoning='[oracle] allow', verdict='allow',content=None)
	def adjudicate_result(self, conversation: Conversation, action: AttackerAction, raw_result: str) -> DefenderDecision:
		return DefenderDecision(reasoning='[oracle] redact', verdict='redact',content=self._scrub(raw_result))
	def respond(self, conversation: Conversation, action: AttackerAction) -> DefenderDecision:
		return DefenderDecision(reasoning='[oracle] respond', verdict='respond',content='(ok)')