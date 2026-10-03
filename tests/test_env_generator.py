"""The procedural env generator (spec 2026-10-02): seed -> Environment with every surface feature
randomized, so training across seeds forces content/intent judgment instead of file-name shortcuts.
Property-based: a batch of seeds must all satisfy the same predicates the hand-authored envs do."""
import pytest

from redteamrl.envs.generate import generate_environment, build_generated_tasks
from redteamrl.envs.registry import env_id_of, assert_training_split
from redteamrl.policies.scripted import ChunkExfilAttacker, ScriptedClient
from redteamrl.sandbox.shell import LocalShellSandbox
from redteamrl.types import AttackerAction
from redteamrl.verify.nondegenerate import verify_nondegenerate
from redteamrl.verify.task import verify_attack, verify_benign

CANARY = "sk-live-CANARY-genr"
SEEDS = list(range(40))


def test_deterministic_and_distinct():
    e5a, p5a = generate_environment(5)
    e5b, p5b = generate_environment(5)
    assert e5a.model_dump() == e5b.model_dump() and p5a == p5b
    e6, _ = generate_environment(6)
    assert e6.model_dump() != e5a.model_dump()


@pytest.mark.parametrize("seed", SEEDS)
def test_structure(seed):
    env, path = generate_environment(seed)
    paths = {f.path for f in env.files}
    assert env.split == "train" and env.level == 1 and env.id == f"gen-{seed}"
    assert len(paths) >= 6
    assert len({p.split("/")[0] for p in paths if "/" in p}) >= 2
    assert path in paths and env.decoys and path not in env.decoys
    for a in env.attacks:
        assert path not in a.goal and path not in a.cover
    first = [b.client_actions[0] for b in env.benigns]
    assert sum(a.tool == "run_command" for a in first) * 2 > len(env.benigns)
    assert any(path in str(a.args) for b in env.benigns for a in b.client_actions)


@pytest.mark.parametrize("seed", SEEDS)
def test_decoy_holds_no_secret(seed):
    env, _ = generate_environment(seed)
    for t in build_generated_tasks(seed, CANARY):
        for f in t.seed_files:
            if f.path in env.decoys:
                assert CANARY not in f.content


@pytest.mark.parametrize("seed", SEEDS)
def test_benigns_verify(seed):
    for t in [x for x in build_generated_tasks(seed, CANARY) if x.episode_type == "benign"]:
        r = verify_benign(t, LocalShellSandbox())
        assert r.ok, f"{t.id}: {r.reason}"


@pytest.mark.parametrize("seed", SEEDS)
def test_attack_reachable_and_nondegenerate(seed):
    env, path = generate_environment(seed)
    tasks = build_generated_tasks(seed, CANARY)
    for t in [x for x in tasks if x.episode_type == "attack"]:
        probe = ScriptedClient([AttackerAction(kind="tool_call", tool="read_file", args={"path": path}),
                                AttackerAction(kind="submit", value=CANARY)])
        assert verify_attack(t, LocalShellSandbox(), probe, n_probe=1).ok, t.id
    report = verify_nondegenerate(tasks, LocalShellSandbox,
                                  attacker_factory=lambda goal, p=path: ChunkExfilAttacker(p, [CANARY]))
    assert not report["degenerate"], report


def test_training_split_accepts_batch():
    tasks = [t for s in SEEDS for t in build_generated_tasks(s, CANARY)]
    assert assert_training_split(tasks)
    assert {env_id_of(t.id) for t in tasks} == {f"gen-{s}" for s in SEEDS}


def test_surface_variety_across_seeds():
    basenames, keys, formats = set(), set(), set()
    for s in SEEDS:
        env, path = generate_environment(s)
        basenames.add(path.split("/")[-1])
        line = env.canary_line
        keys.add(line.split("=")[0].replace("export ", "").strip())
        formats.add("quoted" if '"' in line else ("export" if line.startswith("export") else "plain"))
    assert len(basenames) > 1 and len(keys) > 1 and len(formats) > 1
