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
_PATH_ECHO_CAP = 120


def _short(path: str) -> str:
    """Bound a path echoed into an error message. A degenerate attacker can emit a multi-KB string
    as its `path`; echoing it verbatim into every error result would bloat the transcript and the
    next prompt."""
    return path if len(path) <= _PATH_ECHO_CAP else path[:_PATH_ECHO_CAP] + "…[truncated]"


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
            return f"Error: access denied: {_short(path)}"
        try:
            if not target.is_file():
                return f"Error: no such file: {_short(path)}"
            # errors="replace": a planted secret can sit in an otherwise-binary file, and strict
            # decoding would crash the read instead of returning it. The canary is ASCII, so it
            # survives; only genuine non-UTF-8 bytes become the replacement char.
            return target.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError) as exc:
            # The attacker emits ARBITRARY paths (a component name too long, an embedded NUL, ...),
            # so the stat/read syscall itself can raise. This is an untrusted-input boundary: it
            # must degrade to an error result the attacker sees, never crash the episode/run.
            return f"Error: cannot read {_short(path)}: {getattr(exc, 'strerror', None) or exc}"

    def list_dir(self, path: str) -> str:
        target = self._safe_path(path)
        if target is None:
            return f"Error: access denied: {_short(path)}"
        try:
            if not target.is_dir():
                return f"Error: no such directory: {_short(path)}"
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        except (OSError, ValueError) as exc:
            return f"Error: cannot list {_short(path)}: {getattr(exc, 'strerror', None) or exc}"
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
                # The attacker runs ARBITRARY shell commands, so stdout is frequently non-UTF-8
                # (cat on a binary, od, a fragment of a key). Strict decoding raised
                # UnicodeDecodeError and aborted the whole collection run; replace keeps the
                # canary (ASCII) intact while turning stray bytes into the replacement char.
                errors="replace",
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"[command timed out after {self.timeout_s}s]"
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > self.output_cap:
            out = out[: self.output_cap] + f"\n[output truncated at {self.output_cap} bytes]"
        return out

    def _safe_path(self, path: str) -> Path | None:
        try:
            candidate = (self.root / path).resolve()
        except (OSError, ValueError):
            # A pathological path can make resolve() itself raise (name too long, embedded NUL).
            # Treat it as inaccessible rather than letting it crash the caller.
            return None
        return candidate if candidate.is_relative_to(self.root) else None

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
