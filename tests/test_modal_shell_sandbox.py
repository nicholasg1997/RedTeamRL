from types import SimpleNamespace

import pytest

from redteamrl.sandbox.modal_shell import ModalShellSandbox
from redteamrl.types import AttackerAction, TaskSpec


class _Stream:
    def __init__(self, value=""):
        self.value = value

    def read(self):
        return self.value


class _Process:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = _Stream(stdout)
        self.stderr = _Stream(stderr)
        self.returncode = returncode

    def wait(self):
        return self.returncode


class _Filesystem:
    def __init__(self):
        self.writes = {}

    def write_text(self, content, path):
        self.writes[path] = content


class _FakeSandbox:
    def __init__(self):
        self.filesystem = _Filesystem()
        self.exec_calls = []
        self.terminated = False
        self.detached = False

    def exec(self, *argv, **kwargs):
        self.exec_calls.append((argv, kwargs))
        return _Process(stdout="ok")

    def terminate(self, wait=False):
        self.terminated = True

    def detach(self):
        self.detached = True


def _spec():
    return TaskSpec(
        id="test-benign",
        episode_type="benign",
        enabled_tools=["run_command"],
        request="read config",
        required=["ok"],
        forbidden=["secret"],
        seed_files=[{"path": "app/config.py", "content": "ok secret"}],
        client_actions=[AttackerAction(kind="tool_call", tool="run_command", args={"command": "true"})],
    )


def test_reset_creates_network_blocked_unmounted_sandbox_and_writes_seed_files():
    created = []

    def create(*args, **kwargs):
        created.append((args, kwargs))
        return _FakeSandbox()

    env = ModalShellSandbox(app="APP", image="IMAGE", sandbox_create=create)
    env.reset(_spec())

    args, kwargs = created[0]
    assert args == ("sleep", "infinity")
    assert kwargs["block_network"] is True
    assert kwargs["cpu"] == (1.0, 2.0)
    assert kwargs["memory"] == (512, 1024)
    assert "volumes" not in kwargs and "secrets" not in kwargs
    assert env._sandbox.filesystem.writes["/workspace/app/config.py"] == "ok secret"


@pytest.mark.parametrize("path", ["../runs", "/runs", "app/../../runs"])
def test_direct_file_tools_reject_paths_outside_workspace(path):
    env = ModalShellSandbox(app="APP", image="IMAGE", sandbox_create=lambda *a, **k: _FakeSandbox())
    env.reset(_spec())
    assert "access denied" in env.read_file(path)
    assert "access denied" in env.list_dir(path)


def test_model_command_is_written_then_executed_inside_sandbox_workdir():
    env = ModalShellSandbox(app="APP", image="IMAGE", sandbox_create=lambda *a, **k: _FakeSandbox())
    env.reset(_spec())
    assert env.run_command("cat app/config.py") == "ok"
    assert env._sandbox.filesystem.writes["/tmp/redteamrl-command.sh"] == "cat app/config.py"
    argv, kwargs = env._sandbox.exec_calls[-1]
    assert argv[:2] == ("bash", "-c")
    assert kwargs["workdir"] == "/workspace"


def test_close_terminates_sandbox():
    env = ModalShellSandbox(app="APP", image="IMAGE", sandbox_create=lambda *a, **k: _FakeSandbox())
    env.reset(_spec())
    sandbox = env._sandbox
    env.close()
    assert sandbox.terminated is True
    assert sandbox.detached is True
    assert env._sandbox is None


def test_exit_metadata_preserves_error_output_and_resets_for_next_command():
    fake = _FakeSandbox()
    env = ModalShellSandbox(app="APP", image="IMAGE", sandbox_create=lambda *a, **k: fake)
    env.reset(_spec())
    fake.exec = lambda *a, **k: _Process(stderr="command failed", returncode=7)
    assert env.run_command("exit 7") == "command failed"
    assert env.last_execution == {"status":"nonzero_exit", "returncode":7, "output_truncated":False}
    fake.exec = lambda *a, **k: _Process(stdout="ok", returncode=0)
    assert env.read_file("app/config.py") == "ok"
    assert env.last_execution["returncode"] == 0


def test_timeout_and_output_cap_have_structured_metadata():
    import modal
    fake = _FakeSandbox()
    env = ModalShellSandbox(app="APP", image="IMAGE", output_cap=3, sandbox_create=lambda *a, **k: fake)
    env.reset(_spec())
    fake.exec = lambda *a, **k: _Process(stdout="12345")
    assert "truncated" in env.run_command("noisy")
    assert env.last_execution["output_truncated"]
    def timeout(*args, **kwargs):
        raise modal.exception.TimeoutError("test")
    fake.exec = timeout
    assert "timed out" in env.run_command("slow")
    assert env.last_execution["status"] == "timeout"
