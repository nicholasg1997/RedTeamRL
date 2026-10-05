import glob, os, datetime
from redteamrl.tasks.schema import load_task_spec
from redteamrl.policies.mlx_client import make_mlx_generate
from redteamrl.policies.prompted import PromptedAttacker, PromptedDefender
from redteamrl.sandbox.local import LocalSandbox
from redteamrl.eval.harness import run_eval

ATTACKER_MODEL = "mlx-community/Qwen3-8B-4bit-DWQ-053125"
DEFENDER_MODEL = "mlx-community/Qwen3-4B-4bit"
v1 = os.path.join(os.path.dirname(__file__), "..", "data", "tasks", "v1")

def main(n_rollouts: int = 5) -> None:
	tasks = [load_task_spec(path) for path in glob.glob(os.path.join(v1, "tier0-*.yaml"))]
	gen_atk = make_mlx_generate(ATTACKER_MODEL)
	gen_def = make_mlx_generate(DEFENDER_MODEL)
	os.makedirs('eval-runs', exist_ok=True)
	log_path = (os.path.join('eval-runs', f"baseline-{datetime.date.today().isoformat()}.jsonl"))

	report = run_eval(
		tasks=tasks,
		attacker_factory=lambda goal, cover=None: PromptedAttacker(generate=gen_atk, goal=goal),
		defender=PromptedDefender(generate=gen_def),
		sandbox_factory=LocalSandbox,
		n_rollouts=n_rollouts,
		log_path=log_path,
		attacker_id=ATTACKER_MODEL,
		defender_id=DEFENDER_MODEL,
	)
	print(report.summary())
	print(f"\nrun-log: {log_path}")

if __name__ == "__main__":
	main()