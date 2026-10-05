from __future__ import annotations

_PLACEHOLDER = "/sandbox"


def normalize_sandbox_paths(text: str, sandbox) -> str:
    """Replace a sandbox's absolute tmpdir root with a stable placeholder so tool results are
    identical across episodes that live in different tmpdirs. `run_command` output (`pwd`,
    tracebacks) is the only tool text that embeds the absolute root; read_file/list_dir echo the
    relative path argument, so this is a no-op for them. Reward is unaffected: the canary never
    contains the sandbox path, so egress substring matching is byte-identical before and after."""
    if not text:
        return text
    root = getattr(sandbox, "root", None)
    if root is None:
        return text
    return text.replace(str(root), _PLACEHOLDER)
