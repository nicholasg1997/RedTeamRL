from redteamrl.types import AttackerAction


class ScriptedClient:
	def __init__(self, actions: list[AttackerAction]):
		self._actions = actions
		self._i = 0

	def act(self, conversation: list[dict]) -> AttackerAction:
		if self._i >= len(self._actions):
			return AttackerAction(kind="message", text="(done)")
		action = self._actions[self._i]
		self._i += 1
		return action