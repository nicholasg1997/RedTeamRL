import sys, glob, os, datetime, yaml
from collections import defaultdict
from redteamrl.tasks.schema import load_task_spec
from redteamrl.sandbox.local import LocalSandbox
from redteamrl.verify.task import verify_benign, verify_attack

v1 = os.path.join(os.path.dirname(__file__), "..", "data", "tasks", "v1")
GEN = "mlx-community/Qwen3-8B-4bit-DWQ-053125"

def scenario_stem(task_id: str) -> str:
	parts = task_id.split("-")
	if parts[-1].isdigit():
		parts = parts[:-1]
	return "-".join(parts[:-1])

def main(probe: bool):
	paths = sorted(glob.glob(os.path.join(v1, "tier0-*.yaml")))
	specs = [load_task_spec(path) for path in paths]

	attacker_factory = None
	if probe:
		from redteamrl.policies.mlx_client import make_mlx_generate
		from redteamrl.policies.prompted import PromptedAttacker
		gen = make_mlx_generate(GEN)
		attacker_factory = lambda goal: PromptedAttacker(gen, goal)

	scenarios = defaultdict(lambda: {"attack": None, "benign": [], "attack_leak_rate": None, "benign_gate": "pass"})
	for spec in specs:
		stem = scenario_stem(spec.id)
		if spec.episode_type == "attack":
			scenarios[stem]['attack'] = spec.id
			if probe:
				r = verify_attack(spec, LocalSandbox(), attacker_factory(spec.goal), n_probe=5)
				scenarios[stem]['attacker_leak_rate'] = r.leak_rate
				assert r.ok, f"{spec.id}: attack not winnable by probe ({r.reason})"

		else:
			scenarios[stem]['benign'].append(spec.id)
			r = verify_benign(spec, LocalSandbox())
			assert r.ok, f"{spec.id}: benign not completable ({r.reason})"

	manifest = {
		"version": "v1",
		"generated": datetime.date.today().isoformat(),
		"generator": GEN,
		"probe_ran": probe,
		"scenarios": [{"stem": k, **v} for k, v in sorted(scenarios.items())],
	}

	with open(os.path.join(v1, "manifest.yaml"), "w") as f:
		yaml.dump(manifest, f, sort_keys=False)

	print(f"Froze {len(specs)} tasks across {len(scenarios)} scenarios to {v1}/manifest.yaml")


if __name__ == "__main__":
	main(probe="--no-probe" not in sys.argv)