import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import defaultdict
from typing import Callable
from pydantic import BaseModel
from tqdm import tqdm

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
		sandbox_factory: Callable[[], Sandbox],
		n_rollouts: int = 5,
		max_turns: int = 8,
		log_path: str | Path | None = None,
		attacker_id: str = "",
		defender_id: str = "",
		max_concurrency: int = 1,
) -> EvalReport:
	work = [(task, rollout) for task in tasks for rollout in range(n_rollouts)]
	log_file = open(log_path, "w") if log_path else None
	log_lock = threading.Lock()

	def run_one(item: tuple[TaskSpec, int]):
		task, rollout = item
		if task.episode_type == "attack":
			agent, turns = attacker_factory(task.goal, task.request), max_turns
		else:
			agent, turns = ScriptedClient(task.client_actions), len(task.client_actions)
		sandbox = sandbox_factory()
		try:
			result = run_episode(task, agent, defender, sandbox, max_turns=turns)
		finally:
			sandbox.close()
		if log_file:
			line = json.dumps({
				"task_id": task.id, "episode_type": task.episode_type,
				"rollout": rollout, "result": result.model_dump(),
			})
			with log_lock:
				log_file.write(line + "\n")
		return task.id, task.episode_type, result

	attack: dict[str, list[EpisodeResult]] = defaultdict(list)
	benign: dict[str, list[EpisodeResult]] = defaultdict(list)
	try:
		with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
			for tid, etype, result in tqdm(pool.map(run_one, work), total=len(work)):
				(attack if etype == "attack" else benign)[tid].append(result)
	finally:
		if log_file:
			log_file.close()

	flat_attack = [r for results in attack.values() for r in results]
	flat_benign = [r for results in benign.values() for r in results]

	return EvalReport(
		n_rollouts=n_rollouts, max_turns=max_turns, attacker_id=attacker_id, defender_id=defender_id,
		overall_attack=AttackMetrics.from_results(flat_attack),
		overall_benign=BenignMetrics.from_results(flat_benign),
		attack_by_task={tid: AttackMetrics.from_results(rs) for tid, rs in attack.items()},
		benign_by_task={tid: BenignMetrics.from_results(rs) for tid, rs in benign.items()},
	)