"""Hashing stays inside the same network-blocked container as all replayed commands."""
import inspect
import re

from redteamrl.harness.restart import sandbox_content_hash
from redteamrl.sandbox.modal_shell import ModalShellSandbox


class RestartSandbox(ModalShellSandbox):
    def state_hash(self):
        script = "import os, hashlib\n" + inspect.getsource(sandbox_content_hash)
        script += '\nprint(sandbox_content_hash("/workspace"))\n'
        process = self._require_sandbox().exec("python", "-c", script,
                                              workdir="/workspace", timeout=20)
        output, error = process.stdout.read().strip(), process.stderr.read()
        if process.wait() != 0 or not re.fullmatch(r"[0-9a-f]{64}", output):
            raise RuntimeError("Could not hash replayed workspace: " + error[:200])
        return output
