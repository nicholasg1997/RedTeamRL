import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
	post_completion_turns: int | None = None
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
			f"  BENIGN  completion={b.completion_rate:.1%}  over_refusal={b.over_refusal_rate:.1%}  "
			f"benign_leak={b.benign_leak_rate:.1%}  pre={b.leak_before_completion_rate:.1%}  "
			f"at={b.leak_at_completion_rate:.1%}  post={b.post_completion_leak_rate:.1%}  "
			f"avg_steps={b.avg_steps_to_completion}",
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
		benign_agent_factory: Callable[[str], AttackerPolicy] | None = None,
		task_transform: Callable[[TaskSpec, int], TaskSpec] | None = None,
		resume_dir: str | Path | None = None,
		checkpoint_callback: Callable[[], None] | None = None,
		checkpoint_every: int = 10,
		post_completion_turns: int | None = None,
) -> EvalReport:
	if max_concurrency < 1:
		raise ValueError("max_concurrency must be at least 1")
	if checkpoint_every < 1:
		raise ValueError("checkpoint_every must be at least 1")

	work = [
		(index, task, rollout)
		for index, (task, rollout) in enumerate(
			(task, rollout) for task in tasks for rollout in range(n_rollouts)
		)
	]
	resume_path = Path(resume_dir) if resume_dir is not None else None
	if resume_path is not None:
		resume_path.mkdir(parents=True, exist_ok=True)

	def checkpoint_path(index: int) -> Path:
		assert resume_path is not None
		return resume_path / f"{index:06d}.json"

	def load_checkpoint(index: int, base_task: TaskSpec, rollout: int) -> dict | None:
		if resume_path is None:
			return None
		path = checkpoint_path(index)
		if not path.exists():
			return None
		try:
			with path.open() as f:
				row = json.load(f)
			if (
				not isinstance(row, dict)
				or row.get("schema_version") != 2
				or row.get("work_index") != index
				or row.get("base_task_id") != base_task.id
				or row.get("episode_type") != base_task.episode_type
				or row.get("rollout") != rollout
				or row.get("post_completion_turns_config") != post_completion_turns
			):
				raise ValueError("checkpoint identity does not match the requested evaluation")
			EpisodeResult.model_validate(row["result"])
		except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
			raise RuntimeError(
				f"Invalid or stale evaluation checkpoint {path}; "
				"remove the evaluation checkpoint directory before retrying."
			) from exc
		return row

	def write_checkpoint(index: int, row: dict) -> None:
		if resume_path is None:
			return
		path = checkpoint_path(index)
		tmp = Path(f"{path}.tmp")
		with tmp.open("w") as f:
			json.dump(row, f)
		os.replace(tmp, path)

	def run_one(item: tuple[int, TaskSpec, int]):
		index, base_task, rollout = item
		task = task_transform(base_task, rollout) if task_transform else base_task
		if task.episode_type == "attack":
			agent, turns = attacker_factory(task.goal, task.request), max_turns
		elif benign_agent_factory is not None:
			agent, turns = benign_agent_factory(task.request), max_turns
		else:
			agent, turns = ScriptedClient(task.client_actions), len(task.client_actions)
		sandbox = sandbox_factory()
		try:
			result = run_episode(
				task,
				agent,
				defender,
				sandbox,
				max_turns=turns,
				post_completion_turns=post_completion_turns,
			)
		finally:
			sandbox.close()
		row = {
			"schema_version": 2,
			"work_index": index,
			"base_task_id": base_task.id,
			"task_id": task.id,
			"episode_type": task.episode_type,
			"rollout": rollout,
			"post_completion_turns_config": post_completion_turns,
			"result": result.model_dump(),
		}
		write_checkpoint(index, row)
		return index, row

	completed: dict[int, dict] = {}
	pending: list[tuple[int, TaskSpec, int]] = []
	for item in work:
		index, base_task, rollout = item
		row = load_checkpoint(index, base_task, rollout)
		if row is None:
			pending.append(item)
		else:
			completed[index] = row

	progress = tqdm(total=len(work), initial=len(completed))
	if completed:
		progress.set_description(f"eval (resumed {len(completed)})")
	try:
		with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
			# Keep the worker pool full independently of the persistence cadence. Each worker writes
			# a unique checkpoint through an atomic rename, so committing every N completed episodes
			# is safe even while other workers are still generating.
			futures = [pool.submit(run_one, item) for item in pending]
			since_commit = 0
			for future in as_completed(futures):
				index, row = future.result()
				completed[index] = row
				progress.update(1)
				since_commit += 1
				if (
					resume_path is not None
					and checkpoint_callback is not None
					and since_commit >= checkpoint_every
				):
					checkpoint_callback()
					since_commit = 0
					print(
						f"eval checkpoint: committed {len(completed)}/{len(work)} episodes",
						flush=True,
					)
			if (
				resume_path is not None
				and checkpoint_callback is not None
				and since_commit
			):
				checkpoint_callback()
				print(
					f"eval checkpoint: committed {len(completed)}/{len(work)} episodes",
					flush=True,
				)
	finally:
		progress.close()

	ordered_rows = [completed[index] for index in range(len(work))]
	if log_path is not None:
		path = Path(log_path)
		tmp = Path(f"{path}.tmp")
		with tmp.open("w") as f:
			for row in ordered_rows:
				log_row = {
					"task_id": row["task_id"],
					"episode_type": row["episode_type"],
					"rollout": row["rollout"],
					"result": row["result"],
				}
				f.write(json.dumps(log_row) + "\n")
		os.replace(tmp, path)

	attack: dict[str, list[EpisodeResult]] = defaultdict(list)
	benign: dict[str, list[EpisodeResult]] = defaultdict(list)
	for row in ordered_rows:
		result = EpisodeResult.model_validate(row["result"])
		target = attack if row["episode_type"] == "attack" else benign
		target[row["base_task_id"]].append(result)

	flat_attack = [r for results in attack.values() for r in results]
	flat_benign = [r for results in benign.values() for r in results]

	return EvalReport(
		n_rollouts=n_rollouts, max_turns=max_turns, post_completion_turns=post_completion_turns,
		attacker_id=attacker_id, defender_id=defender_id,
		overall_attack=AttackMetrics.from_results(flat_attack),
		overall_benign=BenignMetrics.from_results(flat_benign),
		attack_by_task={tid: AttackMetrics.from_results(rs) for tid, rs in attack.items()},
		benign_by_task={tid: BenignMetrics.from_results(rs) for tid, rs in benign.items()},
	)
