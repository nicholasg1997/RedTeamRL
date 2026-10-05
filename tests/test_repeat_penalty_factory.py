from redteamrl.train.repeat_penalty import make_repeat_penalty


class _Result:
    def __init__(self, steps):
        self.steps = steps


class _Spec:
    forbidden = ["sk-x"]


def _rep(cmd):
    return {"action": {"kind": "tool_call", "tool": "run_command", "args": {"command": cmd}},
            "returned": "KEY=redacted"}


def test_zero_coef_is_identity():
    t = make_repeat_penalty(0.0)
    assert t(_Result([_rep("a"), _rep("a")]), 1.0, _Spec()) == 1.0


def test_penalty_scales_with_repeat_count():
    t = make_repeat_penalty(0.05)
    # two identical no-progress reads -> one repeat -> -0.05
    assert abs(t(_Result([_rep("a"), _rep("a")]), 0.0, _Spec()) - (-0.05)) < 1e-9
