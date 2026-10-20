"""Both drivers append exactly one metrics row per iteration, in the right place.

The drivers need Modal to import, so the wiring is pinned at the source level (same pattern as
test_v5_finalize.py's driver test). The `print(f"iter ...")` lines are guarded too: metrics are
additive, the human-readable log must not change."""
import ast

import pytest


def _loop(path):
    src = open(path).read()
    tree = ast.parse(src)
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and getattr(n.target, "id", None) == "it"
             and isinstance(n.iter, ast.Call) and getattr(n.iter.func, "id", None) == "range"]
    assert len(loops) == 1
    return src, loops[0]


def _log_calls(loop):
    return [n for n in ast.walk(loop) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "log"
            and getattr(n.func.value, "id", None) == "metrics_log"]


def _last_commit_line(loop):
    commits = [n.lineno for n in loop.body if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == "commit"]
    assert commits
    return commits[-1]


def _calls_named(loop, name):
    return [n for n in ast.walk(loop) if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == name]


def _kw(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


def test_attacker_driver_logs_metrics_once_per_iteration():
    src, loop = _loop("scripts/train_attacker.py")
    assert 'MetricsLog(os.path.join(ckpt_root, "metrics.jsonl"))' in src
    calls = _log_calls(loop)
    assert len(calls) == 1
    call = calls[0]
    assert call.args[0].id == "it" and call.args[1].value == "attacker"
    assert getattr(_kw(call, "cum_gen"), "id", None) == "cumulative_gen_tokens"
    heldout = _calls_named(loop, "_run_heldout_eval")
    assert heldout and heldout[0].lineno < call.lineno < _last_commit_line(loop)
    built = _calls_named(loop, "attacker_metrics")
    assert len(built) == 1 and _kw(built[0], "heldout") is not None
    rollout_min = [n.lineno for n in ast.walk(loop) if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "rollout_min" for t in n.targets)]
    assert rollout_min and rollout_min[0] < call.lineno


def test_defender_driver_logs_metrics_once_per_iteration():
    src, loop = _loop("scripts/train_defender.py")
    assert 'MetricsLog(os.path.join(CKPT_ROOT, "metrics.jsonl"))' in src
    calls = _log_calls(loop)
    assert len(calls) == 1
    call = calls[0]
    assert call.args[0].id == "it" and call.args[1].value == "defender"
    cum_gen = _kw(call, "cum_gen")
    assert isinstance(cum_gen, ast.Constant) and cum_gen.value is None
    assert call.lineno < _last_commit_line(loop)
    built = _calls_named(loop, "defender_metrics")
    assert len(built) == 1 and getattr(_kw(built[0], "decomp"), "id", None) == "d"


@pytest.mark.parametrize("path,prefixes", [
    ("scripts/train_attacker.py", ["alloc", "rollout", "v5", "v6", "replace", "tokens", "tactics", "update", "heldout"]),
    ("scripts/train_defender.py", ["rollout", "tactics", "update"]),
])
def test_print_lines_untouched(path, prefixes):
    src = open(path).read()
    for prefix in prefixes:
        assert f'print(f"iter {{it:3d}} {prefix}' in src, prefix


@pytest.mark.parametrize("path,builder", [
    ("scripts/train_attacker.py", "attacker_metrics"),
    ("scripts/train_defender.py", "defender_metrics"),
])
def test_metrics_build_and_log_are_guarded(path, builder):
    # The writer never raises, but the BUILDER runs in the driver: a missing source key would raise
    # KeyError and kill the run. Both calls must sit inside a try whose handler catches Exception.
    _, loop = _loop(path)
    guarded = [n for n in ast.walk(loop) if isinstance(n, ast.Try)
               and any(getattr(h.type, "id", None) == "Exception" for h in n.handlers)]
    assert guarded, "no try/except Exception in the loop"
    inside = {id(c) for t in guarded for stmt in t.body for c in ast.walk(stmt)}
    assert any(id(c) in inside for c in _calls_named(loop, builder))
    assert any(id(c) in inside for c in _log_calls(loop))
