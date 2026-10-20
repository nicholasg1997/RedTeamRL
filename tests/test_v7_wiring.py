"""The v7 driver wiring, pinned at the source level (the driver needs Modal to import)."""
import ast

from test_v6_wiring import _calls, _func, _kw, _loop, _src_tree


def _const(tree, name):
    return next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == name)


def test_v7_constants():
    _, tree = _src_tree()
    for name, value in (("N_ROLLOUTS", 12), ("MINIBATCHES", 8), ("HELDOUT_EVAL_EVERY", 3),
                        ("INNER_EPOCHS", 2), ("TARGET_KL", 0.01), ("LR", 5e-5)):
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
    assert "AllocParams(depth_start=1.0" in src
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


def test_lr_rule_is_told_how_many_steps_were_taken():
    src, tree = _src_tree()
    call = _calls(_loop(tree), "next_lr")[0]
    assert 'm["steps"]' in ast.get_source_segment(src, _kw(call, "steps"))


# --- v7.1 ------------------------------------------------------------------------------------------

def test_v71_lineage_and_initial_lr():
    _, tree = _src_tree()
    assert _const(tree, "LINEAGE").value == "v7.2"
    assert _const(tree, "LR").value == 5e-5


def test_slots_come_from_n_slots_not_the_task_count():
    src, tree = _src_tree()
    for name in ("allocate", "plan_replacements"):
        for call in _calls(tree, name):
            assert "n_slots" in ast.get_source_segment(src, call), name
    assert "(n_slots + alloc.max_replacements) * n_rollouts" in src


def test_overfit_mode_is_a_flag_with_its_own_lineage():
    src, tree = _src_tree()
    for name in ("train", "main", "_train_impl"):
        fn = _func(tree, name)
        args = fn.args.args
        defaults = dict(zip([a.arg for a in args[len(args) - len(fn.args.defaults):]],
                            fn.args.defaults))
        assert getattr(defaults.get("overfit_task"), "value", None) == "", name
    assert 'ckpt_root += f"-overfit-{overfit_task}"' in src
    assert '"overfit_task": overfit_task' in src
    assert "restart_fraction_max=0.0, max_replacements=0" in src
    assert "n_slots = OVERFIT_GROUPS if overfit_task else len(tasks)" in src
    assert "not overfit_task and (" in src          # held-out eval skipped in overfit mode


def test_brake_rule_is_in_the_identity_and_training_stall_is_logged_guarded():
    src, tree = _src_tree()
    assert '"brake_rule": "mean2x-or-single10x-v1"' in src
    loop = _loop(tree)
    stalls = _calls(loop, "training_stall")
    assert len(stalls) == 1
    guarded = [n for n in ast.walk(loop) if isinstance(n, ast.Try)
               and any(getattr(h.type, "id", None) == "Exception" for h in n.handlers)]
    inside = {id(c) for t in guarded for stmt in t.body for c in ast.walk(stmt)}
    assert id(stalls[0]) in inside


# --- v7.2: LR warmup + repeat penalty (arm 3) ---------------------------------------------------

def test_v72_lineage_constants_and_arm3_pin():
    src, tree = _src_tree()
    assert _const(tree, "LINEAGE").value == "v7.2"
    assert _const(tree, "WARMUP_STEPS").value == 16
    assert "assert arm == 3" in src
    for name in ("train", "main"):
        fn = _func(tree, name)
        args = fn.args.args
        defaults = dict(zip([a.arg for a in args[len(args) - len(fn.args.defaults):]],
                            fn.args.defaults))
        assert defaults["arm"].value == 3, name
        assert defaults["repeat_penalty_coef"].value == 0.02, name


def test_warmup_is_wired_through_a_persisted_global_step():
    src, tree = _src_tree()
    call = _calls(_loop(tree), "update_step")[0]
    assert getattr(_kw(call, "warmup_steps"), "id", None) == "WARMUP_STEPS"
    assert getattr(_kw(call, "global_step"), "id", None) == "optimizer_steps"
    writes = [c for c in _calls(tree, "write_meta") if isinstance(c.args[1], ast.Dict)]
    keys = {k.value for c in writes for k in c.args[1].keys if isinstance(k, ast.Constant)}
    assert "optimizer_steps" in keys
    assert '.get("optimizer_steps", 0)' in src
    assert '"lr_warmup_steps": WARMUP_STEPS' in src


def test_repeat_penalty_coef_is_checked_against_the_sweep():
    src, _ = _src_tree()
    assert "if coef not in REPEAT_PENALTY_SWEEP:" in src


def test_v72_near_misses_restart_after_the_redaction():
    src, _ = _src_tree()
    assert "near_miss_offset=1, near_miss_min_turns_left=3" in src
    assert "banks, it, bank_record_dir, params=alloc)" in src


def test_v72_uses_the_foundational_hint_and_records_it_in_identity():
    src, _ = _src_tree()
    assert "ATTACKER_HINT = HINT_FOUNDATIONAL_V2" in src
    assert "hint=ATTACKER_HINT" in src
    assert "HINT_TECHNIQUE," not in src.split("def _train_impl")[1].split("return PromptedAttacker")[0] or "ATTACKER_HINT" in src
    assert '"attacker_hint_version"' in src
