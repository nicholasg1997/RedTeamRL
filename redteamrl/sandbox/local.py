from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from redteamrl.types import TaskSpec

class LocalSandbox:

	def __init__(self):
		self.root = Path(tempfile.mkdtemp(prefix="redteam-sandbox-")).resolve()

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

	def _safe_path(self, path: str) -> Path | None:
		candidate = (self.root / path).resolve()
		return candidate if candidate.is_relative_to(self.root) else None

	def close(self):
		shutil.rmtree(self.root, ignore_errors=True)

