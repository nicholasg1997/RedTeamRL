"""Isolated Modal-backed shell environment for model-generated commands.

Unlike ``LocalShellSandbox``, this backend is a real security boundary: commands run in a
separate Modal Sandbox with no secrets or volumes mounted and outbound networking disabled.
The GPU training function remains only the orchestrator and never executes attacker text.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Callable

import modal

from redteamrl.types import TaskSpec


_TIMEOUT_S = 10
_OUTPUT_CAP = 4096
_EPISODE_TIMEOUT_S = 30 * 60
_WORKDIR = "/workspace"


class ModalShellSandbox:
    """One disposable, network-blocked Modal Sandbox per episode.

    ``sandbox_create`` is injectable so lifecycle and path behavior can be tested locally
    without contacting Modal. Production callers should leave it as ``None``.
    """

    def __init__(
        self,
        app,
        image,
        timeout_s: int = _TIMEOUT_S,
        output_cap: int = _OUTPUT_CAP,
        episode_timeout_s: int = _EPISODE_TIMEOUT_S,
        sandbox_create: Callable[..., Any] | None = None,
    ):
        self.app = app
        self.image = image
        self.timeout_s = timeout_s
        self.output_cap = output_cap
        self.episode_timeout_s = episode_timeout_s
        self._sandbox_create = sandbox_create or modal.Sandbox.create
        self._sandbox = None

    def reset(self, spec: TaskSpec) -> None:
        """Start from a fresh container and plant this episode's synthetic files."""
        self.close()
        self._sandbox = self._sandbox_create(
            "sleep",
            "infinity",
            app=self.app,
            image=self.image,
            workdir=_WORKDIR,
            block_network=True,
            timeout=self.episode_timeout_s,
            # A 27B reasoning turn can leave the environment untouched for several minutes.
            # Keep the isolated container alive for the episode rather than treating that as idle.
            idle_timeout=self.episode_timeout_s,
            # Explicit upper bounds prevent a hostile command from bursting into an expensive
            # amount of CPU or memory. Modal interprets these tuples as (request, limit).
            cpu=(1.0, 2.0),
            memory=(512, 1024),
        )
        for seed_file in spec.seed_files:
            remote = self._remote_path(seed_file.path)
            if remote is None:
                self.close()
                raise ValueError(f"seed_file path escapes sandbox root: {seed_file.path!r}")
            self._sandbox.filesystem.write_text(seed_file.content, remote)

    def read_file(self, path: str) -> str:
        remote = self._remote_path(path)
        if remote is None:
            return f"Error: access denied: {path}"
        return self._run_argv("head", "-c", str(self.output_cap + 1), "--", remote)

    def list_dir(self, path: str) -> str:
        remote = self._remote_path(path)
        if remote is None:
            return f"Error: access denied: {path}"
        out = self._run_argv("ls", "-1Ap", "--", remote)
        if out.startswith("Error:"):
            return out
        return f"{path}:\n" + (out.rstrip("\n") if out else "(empty)")

    def run_command(self, command: str) -> str:
        """Execute arbitrary attacker text inside the isolated container, never the trainer."""
        sandbox = self._require_sandbox()
        command_path = "/tmp/redteamrl-command.sh"
        sandbox.filesystem.write_text(command, command_path)
        # The outer pipeline is authored by the harness, not the agent. It bounds captured output
        # even if the attacker command is deliberately noisy.
        wrapper = (
            "set -o pipefail; "
            f"bash {command_path} 2>&1 | head -c {self.output_cap + 1}"
        )
        return self._run_argv("bash", "-c", wrapper)

    def close(self) -> None:
        if self._sandbox is not None:
            sandbox = self._sandbox
            self._sandbox = None
            try:
                sandbox.terminate(wait=True)
            finally:
                # Release the client's connection even when termination raises. The remote
                # timeout remains the final cleanup backstop if the worker itself is preempted.
                sandbox.detach()

    def _run_argv(self, *argv: str) -> str:
        sandbox = self._require_sandbox()
        try:
            process = sandbox.exec(*argv, workdir=_WORKDIR, timeout=self.timeout_s)
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            returncode = process.wait()
        except modal.exception.TimeoutError:
            return f"[command timed out after {self.timeout_s}s]"

        output = (stdout or "") + (stderr or "")
        if len(output) > self.output_cap:
            output = output[: self.output_cap] + f"\n[output truncated at {self.output_cap} bytes]"
        if returncode != 0 and not output:
            return f"Error: command exited with status {returncode}"
        return output

    def _require_sandbox(self):
        if self._sandbox is None:
            raise RuntimeError("sandbox has not been reset for an episode")
        return self._sandbox

    @staticmethod
    def _remote_path(path: str) -> str | None:
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            return None
        parts = [part for part in candidate.parts if part not in ("", ".")]
        return _WORKDIR + ("/" + "/".join(parts) if parts else "")
