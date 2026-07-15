import json
from pathlib import Path
from collections import defaultdict
from typing import Callable
from pydantic import BaseModel

from redteamrl.types import TaskSpec
from redteamrl.policies.base import AttackerPolicy, DefenderPolicy
from redteamrl.policies.scripted import ScriptedClient
from redteamrl.sandbox.base import Sandbox
from redteamrl.harness.episode import run_episode, EpisodeResult
from redteamrl.eval.metrics import AttackMetrics, BenignMetrics

class EvalReport(BaseModel):
	n_rollouts: int
	max_turns: int
	attacker_id: str
	defender_id: str
	overall_attack: AttackMetrics
	overall_benign: BenignMetrics
	attack_by_task: dict[str, AttackMetrics]
	benign_by_task: dict[str, BenignMetrics]

	def summary(self) -> str:
		a, b = self.overall_attack, self.overall_benign
		return "\n".join([
			f"Eval  attacker={self.attacker_id or '?'}  defender={self.defender_id or '?'}  n_rollouts={self.n_rollouts}",
			f"  ATTACK  success={a.attack_success_rate:.1%}  defense={1 - a.attack_success_rate:.1%}  avg_steps_to_leak={a.avg_steps_to_leak}",
			f"  BENIGN  completion={b.completion_rate:.1%}  over_refusal={b.over_refusal_rate:.1%}  benign_leak={b.benign_leak_rate:.1%}  avg_steps={b.avg_steps_to_completion}",
		])

def run_eval(
		tasks: list[TaskSpec],
		attacker_factory: Callable[[str], AttackerPolicy],
		defender: DefenderPolicy,
		sandbox: Sandbox,
		n_rollouts: int = 5,
		max_turns: int = 8,
		log_path: str | Path | None = None,
		attacker_id: str = "",
		defender_id: str = "",
) -> EvalReport:
	attack: dict[str, list[EpisodeResult]] = defaultdict(list)
	benign: dict[str, list[EpisodeResult]] = defaultdict(list)
	log_file = open(log_path, "w") if log_path else None
	try:
		for task in tasks:
			for run in range(n_rollouts):
				if task.episode_type == "attack":
					agent, turns = attacker_factory(task.goal), max_turns
				else:
					agent, turns = ScriptedClient(task.client_actions), len(task.client_actions)
				result = run_episode(task, agent, defender, sandbox, max_turns=turns)
				if log_file:
					log_file.write(json.dumps({
						"task_id": task.id, "episode_type": task.episode_type,
						"seed": run, "result": result.model_dump(),
					}) + "\n")
				(attack if task.episode_type == "attack" else benign)[task.id].append(result)
	finally:
		if log_file:
			log_file.close()

	flat_attack = [r for rs in attack.values() for r in rs]
	flat_benign = [r for rs in benign.values() for r in rs]

	return EvalReport(
		n_rollouts=n_rollouts, max_turns=max_turns, attacker_id=attacker_id, defender_id=defender_id,
		overall_attack=AttackMetrics.from_results(flat_attack),
		overall_benign=BenignMetrics.from_results(flat_benign),
		attack_by_task={tid: AttackMetrics.from_results(rs) for tid, rs in attack.items()},
		benign_by_task={tid: BenignMetrics.from_results(rs) for tid, rs in benign.items()},
	)