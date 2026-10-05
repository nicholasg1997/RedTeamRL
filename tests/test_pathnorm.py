from redteamrl.harness.pathnorm import normalize_sandbox_paths


class _FakeSandbox:
    def __init__(self, root):
        self.root = root


def test_absolute_root_is_replaced_with_placeholder():
    sb = _FakeSandbox("/tmp/redteam-shell-abc123")
    out = normalize_sandbox_paths("/tmp/redteam-shell-abc123/config.env\n", sb)
    assert out == "/sandbox/config.env\n"


def test_two_sandboxes_with_different_roots_normalise_to_same_text():
    a = normalize_sandbox_paths("cwd is /tmp/redteam-shell-AAA", _FakeSandbox("/tmp/redteam-shell-AAA"))
    b = normalize_sandbox_paths("cwd is /tmp/redteam-shell-BBB", _FakeSandbox("/tmp/redteam-shell-BBB"))
    assert a == b == "cwd is /sandbox"


def test_missing_root_is_noop():
    class NoRoot:  # ModalShellSandbox-style: no .root
        pass
    assert normalize_sandbox_paths("unchanged /tmp/x", NoRoot()) == "unchanged /tmp/x"


def test_empty_text_is_noop():
    assert normalize_sandbox_paths("", _FakeSandbox("/tmp/redteam-shell-abc")) == ""
