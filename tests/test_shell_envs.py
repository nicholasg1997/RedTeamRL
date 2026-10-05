from redteamrl.envs import build_tasks
from redteamrl.sandbox.shell import LocalShellSandbox
from redteamrl.verify.task import verify_benign
from redteamrl.types import AttackerAction
from redteamrl.policies.scripted import ChunkExfilAttacker
from redteamrl.envs.ops_logs import OPS_LOGS
from redteamrl.envs.analytics import ANALYTICS
from redteamrl.envs.ci_build import CI_BUILD
from redteamrl.verify.nondegenerate import verify_nondegenerate

CANARY = "sk-live-CANARY-0001"


def test_ops_logs_benign_commands_produce_required_and_verify():
    tasks = build_tasks(OPS_LOGS, CANARY, canary_path="app/config.py")
    for t in [x for x in tasks if x.episode_type == "benign"]:
        r = verify_benign(t, LocalShellSandbox())
        assert r.ok, f"{t.id}: {r.reason}"      # command output == required, canary separable


def test_analytics_benign_commands_verify():
    tasks = build_tasks(ANALYTICS, CANARY, canary_path=".env")
    for t in [x for x in tasks if x.episode_type == "benign"]:
        r = verify_benign(t, LocalShellSandbox())
        assert r.ok, f"{t.id}: {r.reason}"


def test_ci_build_benign_commands_verify():
    tasks = build_tasks(CI_BUILD, CANARY, canary_path="deploy/credentials")
    for t in [x for x in tasks if x.episode_type == "benign"]:
        r = verify_benign(t, LocalShellSandbox())
        assert r.ok, f"{t.id}: {r.reason}"



ENVS = [(OPS_LOGS, "app/config.py"), (ANALYTICS, ".env"), (CI_BUILD, "deploy/credentials")]


def test_all_envs_layer1_and_layer2():
    from redteamrl.envs import build_tasks
    for env, path in ENVS:
        tasks = build_tasks(env, CANARY, canary_path=path)
        # Layer 1: benign well-formed
        for t in [x for x in tasks if x.episode_type == "benign"]:
            assert verify_benign(t, LocalShellSandbox()).ok, t.id
        # Layer 2: non-degeneracy — no trivial policy wins
        report = verify_nondegenerate(
            tasks, LocalShellSandbox,
            attacker_factory=lambda goal, p=path: ChunkExfilAttacker(p, [CANARY]))
        assert not report["degenerate"], f"{env.id} degenerate: {report}"