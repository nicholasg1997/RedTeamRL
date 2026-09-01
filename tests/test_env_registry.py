"""The held-out reserve is a rule, not an intention (PLAN.md §2.9).

Three environments have never been trained on -- but by accident, because nobody wired them in,
not because anyone reserved them. One script adding `billing` to a training list would have
evaporated the reserve silently and nothing would have failed. These tests are what makes the
reserve survive a deadline.
"""
import pytest

from redteamrl.envs import build_tasks
from redteamrl.envs.billing import BILLING
from redteamrl.envs.ops_logs import OPS_LOGS
from redteamrl.envs.registry import (
    ALL_ENVIRONMENTS, HELD_OUT_ENVS, TRAIN_ENVS,
    assert_training_split, env_id_of, held_out_tasks,
)


def test_env_id_of_handles_hyphenated_env_ids():
    # four of six env ids contain hyphens, so splitting on "-" would truncate them
    assert env_id_of("ops-logs-attack-exfil") == "ops-logs"
    assert env_id_of("ci-build-benign-region") == "ci-build"
    assert env_id_of("support-portal-attack-token") == "support-portal"
    assert env_id_of("analytics-benign-ttl") == "analytics"


def test_env_id_of_rejects_unparseable_id():
    with pytest.raises(ValueError):
        env_id_of("not-a-task-id")


def test_assert_training_split_rejects_reserved_environment():
    tasks = (build_tasks(OPS_LOGS, "c", canary_path="app/config.py")
             + build_tasks(BILLING, "c", canary_path="app/config.py"))
    with pytest.raises(ValueError, match="billing"):
        assert_training_split(tasks)


def test_assert_training_split_passes_training_only_and_returns_tasks():
    tasks = build_tasks(OPS_LOGS, "c", canary_path="app/config.py")
    assert assert_training_split(tasks) is tasks


def test_every_environment_declares_a_split():
    # the forcing function: a new environment cannot be added without choosing a side
    assert len(ALL_ENVIRONMENTS) == 6
    for env in ALL_ENVIRONMENTS:
        assert env.split in ("train", "held_out"), env.id


def test_reserve_is_non_empty_and_disjoint_from_training():
    assert HELD_OUT_ENVS and TRAIN_ENVS
    assert not ({e.id for e in TRAIN_ENVS} & {e.id for e in HELD_OUT_ENVS})


def test_held_out_tasks_builds_only_reserved_environments():
    reserved = {e.id for e in HELD_OUT_ENVS}
    assert {env_id_of(t.id) for t in held_out_tasks()} == reserved


def test_held_out_canary_lands_in_an_existing_file():
    """build_tasks CREATES canary_path when it does not exist, which would plant the secret in a
    lone unexplained file -- trivially detectable, and not the co-located design these envs use."""
    for env in HELD_OUT_ENVS:
        from redteamrl.envs.registry import CANARY_PATHS
        assert CANARY_PATHS[env.id] in {f.path for f in env.files}, env.id


def test_held_out_tasks_actually_plant_the_secret():
    for task in held_out_tasks(canary="sk-CANARY-probe"):
        assert any("sk-CANARY-probe" in f.content for f in task.seed_files), task.id


# --- the sweep: a new training script cannot forget the guard -------------------------------

# Scripts that legitimately build tasks WITHOUT the training guard, each with a reason. Exempting
# a script is meant to be a deliberate, reviewed act -- that is the whole value of the list.
_EVAL_SCRIPTS = {
    "probe_grpo_adapter.py": "shortcut probes; evaluates held-out and seen envs on purpose",
    "attacker_viability.py": "measures win rate against the training distribution, trains nothing",
    "shell_probe.py": "one-off inference probe",
    "modal_eval_spike.py": "deprecated YAML-path throughput spike",
}


def test_training_scripts_cannot_forget_the_guard():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "scripts"
    offenders = []
    for path in sorted(root.glob("*.py")):
        src = path.read_text()
        if "build_tasks" not in src or path.name in _EVAL_SCRIPTS:
            continue
        if "assert_training_split" not in src:
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} build tasks without assert_training_split. Add the guard, or add the script "
        f"to _EVAL_SCRIPTS with a reason if it genuinely does not train."
    )
