"""The v7 driver wiring, pinned at the source level (the driver needs Modal to import)."""
import ast

from test_v6_wiring import _calls, _func, _kw, _loop, _src_tree


def _const(tree, name):
    return next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == name)


def test_v7_constants():
    _, tree = _src_tree()
    for name, value in (("N_ROLLOUTS", 12), ("MINIBATCHES", 8), ("HELDOUT_EVAL_EVERY", 3),
                        ("INNER_EPOCHS", 2), ("TARGET_KL", 0.01), ("LR", 3e-5)):
        assert getattr(_const(tree, name), "value", None) == value, name
    # Module scope imports only modal (redteamrl is imported inside _train_impl), so the rule
    # object is built there from the module constants.
    assert "lr_rule = LrRule(initial=LR, target_kl=TARGET_KL)" in _src_tree()[0]


def test_identity_carries_the_v7_rules():
    src, _ = _src_tree()
    assert "phase_identity.update(v7_identity(alloc, DEFAULT_MAX_NEAR_MISSES))" in src
    assert "phase_identity.update(lr_rule.as_identity())" in src
    assert '"minibatches": MINIBATCHES' in src
    assert '"heldout_eval_every": HELDOUT_EVAL_EVERY' in src
    assert "AllocParams(depth_start=1.0)" in src
    assert "finalize_v6(" not in src


def test_update_takes_minibatch_steps_with_the_brake_and_feeds_the_lr_rule():
    src, tree = _src_tree()
    loop = _loop(tree)
    updates = _calls(loop, "update_step")
    assert len(updates) == 1
    call = updates[0]
    assert getattr(_kw(call, "minibatches"), "id", None) == "MINIBATCHES"
    assert "lr_rule.target_kl" in ast.get_source_segment(src, _kw(call, "target_kl"))
    assert "LINEAGE" in ast.get_source_segment(src, _kw(call, "shuffle_seed"))
    lrs = _calls(loop, "next_lr")
    assert len(lrs) == 1 and lrs[0].lineno > call.lineno
    assert "param_groups" in ast.get_source_segment(src, loop)


def test_base_episodes_are_banked_before_replacement_planning():
    src, tree = _src_tree()
    loop = _loop(tree)
    first_rollout = _calls(loop, "rollout")[0]
    planned = _calls(loop, "load_or_plan")[0]
    banked = _calls(loop, "bank_episodes")
    assert len(banked) == 1 and first_rollout.lineno < banked[0].lineno < planned.lineno
    assert '["slot"] < len(plans)' in ast.get_source_segment(src, banked[0].args[0])


def test_graduation_and_same_iteration_preference_reach_selection():
    src, tree = _src_tree()
    assert all(getattr(_kw(c, "graduated"), "id", None) == "graduated"
               for c in _calls(tree, "allocate"))
    calls = _calls(tree, "plan_replacements")
    assert calls and all(getattr(_kw(c, "graduated"), "id", None) == "graduated" for c in calls)
    assert all(getattr(_kw(c, "iteration"), "id", None) == "it" for c in calls)


def test_lr_and_graduated_round_trip_through_meta():
    src, tree = _src_tree()
    writes = [c for c in _calls(tree, "write_meta") if isinstance(c.args[1], ast.Dict)]
    keys = {k.value for c in writes for k in c.args[1].keys if isinstance(k, ast.Constant)}
    assert {"lr", "graduated"} <= keys
    assert '.get("lr", lr_rule.initial)' in src
    assert '.get("graduated", [])' in src
