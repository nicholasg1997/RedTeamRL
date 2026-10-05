import contextlib

from redteamrl.policies.scripted import ScriptedClient
from redteamrl.sandbox.local import LocalSandbox
from redteamrl.train.train import _run_one_episode
from redteamrl.types import AttackerAction, TaskSpec
from redteamrl.verify.defenders import NullDefender


def _spec():
    return TaskSpec(
        id="gen-0", episode_type="attack", tier=0,
        enabled_tools=["read_file", "list_dir"],
        goal="find the key", forbidden=["sk-live-xxx"],
        seed_files=[{"path": "a.txt", "content": "nothing here\n"}],
    )


class _NullCapture:
    @contextlib.contextmanager
    def episode_capture(self):
        yield []


def test_reward_transform_is_applied_to_attacker_reward():
    spec = _spec()
    # Scripted attacker repeats the same no-progress read twice.
    agent = ScriptedClient([
        AttackerAction(kind="tool_call", tool="read_file", args={"path": "a.txt"}),
        AttackerAction(kind="tool_call", tool="read_file", args={"path": "a.txt"}),
    ])
    seen = {}

    def transform(result, base_reward, episode_spec):
        seen["base"] = base_reward
        seen["forbidden"] = episode_spec.forbidden[0]   # transform receives the spec
        return base_reward - 1.0   # obvious marker

    out = _run_one_episode(
        spec, 0, lambda s: agent, lambda s: NullDefender(), LocalSandbox,
        capturing_generate=_NullCapture(), max_turns=4, post_completion_turns=None,
        redaction_enforcement="unshielded", trained_side="attacker",
        reward_transform=transform,
    )
    assert out["reward"] == seen["base"] - 1.0
    assert seen["forbidden"] == "sk-live-xxx"
