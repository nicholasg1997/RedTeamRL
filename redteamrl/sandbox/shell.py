"""LocalShellSandbox — a bash-capable sandbox for LOCAL experimentation.

This is the local counterpart to the (Modal) shell sandbox: it runs `run_command`
in a real shell so we can study transform attacks (base64, chunking, grep -o, ...)
without a GPU. It is NOT a security boundary — it runs commands in a subprocess with
`cwd` pinned to a per-episode tmpdir and a scrubbed environment, with a wall-clock
timeout and an output cap. Safe for scripted / controlled agents on a dev machine;
for an untrusted LLM at scale use container isolation (modal.Sandbox) instead.
"""
from __future__ import annotations
import shutil
import subprocess
import tempfile
from pathlib import Path

from redteamrl.types import TaskSpec

_TIMEOUT_S = 10
_OUTPUT_CAP = 4096


class LocalShellSandbox:
    def __init__(self, timeout_s: int = _TIMEOUT_S, output_cap: int = _OUTPUT_CAP):
        self.root = Path(tempfile.mkdtemp(prefix="redteam-shell-")).resolve()
        self.timeout_s = timeout_s
        self.output_cap = output_cap

    def reset(self, spec: TaskSpec) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        for sf in spec.seed_files:
            dest = self._safe_path(sf.path)
            if dest is None:
                raise ValueError(f"seed_file path escapes sandbox root: {sf.path!r}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(sf.content, encoding="utf-8")

    def read_file(self, path: str) -> str:
        target = self._safe_path(path)
        if target is None:
            return f"Error: access denied: {path}"
        if not target.is_file():
            return f"Error: no such file: {path}"
        return target.read_text(encoding="utf-8")

    def list_dir(self, path: str) -> str:
        target = self._safe_path(path)
        if target is None:
            return f"Error: access denied: {path}"
        if not target.is_dir():
            return f"Error: no such directory: {path}"
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return f"{path}:\n" + ("\n".join(entries) if entries else "(empty)")

    def run_command(self, command: str) -> str:
        """Run `command` in a shell rooted at the sandbox. Scrubbed env, timeout, capped output."""
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=self.root,
                env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(self.root)},
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"[command timed out after {self.timeout_s}s]"
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > self.output_cap:
            out = out[: self.output_cap] + f"\n[output truncated at {self.output_cap} bytes]"
        return out

    def _safe_path(self, path: str) -> Path | None:
        candidate = (self.root / path).resolve()
        return candidate if candidate.is_relative_to(self.root) else None

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
