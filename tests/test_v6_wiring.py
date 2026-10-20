"""The v6 driver wiring, pinned at the source level (the driver needs Modal to import)."""
import ast

SRC_PATH = "scripts/train_attacker.py"


def _src_tree():
    src = open(SRC_PATH).read()
    return src, ast.parse(src)


def _loop(tree):
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and getattr(n.target, "id", None) == "it"
             and isinstance(n.iter, ast.Call) and getattr(n.iter.func, "id", None) == "range"]
    assert len(loops) == 1
    return loops[0]


def _calls(node, name):
    return sorted((n for n in ast.walk(node) if isinstance(n, ast.Call)
                   and getattr(n.func, "id", None) == name), key=lambda n: n.lineno)


def _kw(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _func(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_lineage_identity_and_arm_pin():
    src, tree = _src_tree()
    lineage = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", None) == "LINEAGE")
    assert lineage == "v7"          # v7 keeps every v6 mechanism (tests/test_v7_wiring.py)
    assert "phase_identity.update(v7_identity(alloc, DEFAULT_MAX_NEAR_MISSES))" in src
    assert '"n_rollouts": n_rollouts' in src
    assert "v5_identity(" not in src
    assert "assert arm == 2" in src


def test_replacements_are_planned_before_finalize_and_rolled_into_the_same_store():
    src, tree = _src_tree()
    loop = _loop(tree)
    rollouts = _calls(loop, "rollout")
    assert len(rollouts) == 2
    second = rollouts[1]
    assert getattr(_kw(second, "group_plans"), "id", None) == "replacements"
    assert getattr(_kw(second, "episode_store"), "id", None) == "rollout_dir"
    guards = [n for n in ast.walk(loop) if isinstance(n, ast.If)
              and getattr(n.test, "id", None) == "replacements"]
    assert any(second in list(ast.walk(g)) for g in guards)
    planned = _calls(loop, "load_or_plan")
    assert len(planned) == 1 and rollouts[0].lineno < planned[0].lineno < second.lineno
    assert "replacements.json" in ast.get_source_segment(src, planned[0].args[0])
    finalize = _calls(loop, "finalize_v7")
    advantages = _calls(loop, "assign_advantages")
    assert finalize and second.lineno < finalize[0].lineno < advantages[0].lineno
    assert not _calls(loop, "finalize_iteration")


def test_canary_stride_reserves_replacement_slots():
    src, tree = _src_tree()
    body = ast.get_source_segment(src, _func(tree, "episode_task_for"))
    assert "(len(tasks) + alloc.max_replacements) * n_rollouts" in body


def test_depths_round_trip_through_meta():
    src, tree = _src_tree()
    writes = [c for c in _calls(tree, "write_meta") if isinstance(c.args[1], ast.Dict)]
    keys = {k.value for c in writes for k in c.args[1].keys if isinstance(k, ast.Constant)}
    assert "restart_depths" in keys
    assert '.get("restart_depths", {})' in src
    assert all(_kw(c, "depths") is not None for c in _calls(tree, "allocate"))


def test_group_size_is_a_flag():
    src, tree = _src_tree()
    for name in ("train", "main", "_train_impl"):
        fn = _func(tree, name)
        args = fn.args.args
        defaults = dict(zip([a.arg for a in args[len(args) - len(fn.args.defaults):]],
                            fn.args.defaults))
        assert getattr(defaults.get("n_rollouts"), "id", None) == "N_ROLLOUTS", name
    impl = _func(tree, "_train_impl")
    defaults = set(map(id, impl.args.defaults))
    uses = [n for n in ast.walk(impl) if isinstance(n, ast.Name) and n.id == "N_ROLLOUTS"
            and id(n) not in defaults]
    assert not uses, [n.lineno for n in uses]
