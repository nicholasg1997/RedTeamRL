"""The base-policy held-out eval mode, pinned at the source level (the driver needs Modal to import).

`--baseline-eval` measures the UNTRAINED policy on the held-out seeds through the exact code path a
training run's held-out eval uses: the attacker vLLM server starts with a freshly initialised LoRA
(B = 0, so its output is exactly the base model's), and the eval must run before anything could
replace it -- no checkpoint resume, no publish_policy -- and must write no training state.
"""
import ast

SRC_PATH = "scripts/train_attacker.py"


def _src_tree():
    src = open(SRC_PATH).read()
    return src, ast.parse(src)


def _func(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(node, name):
    return sorted((n for n in ast.walk(node) if isinstance(n, ast.Call)
                   and getattr(n.func, "id", None) == name), key=lambda n: n.lineno)


def _baseline_branch(impl):
    branches = [n for n in ast.walk(impl) if isinstance(n, ast.If)
                and getattr(n.test, "id", None) == "baseline_eval"]
    assert len(branches) == 1, "expected exactly one `if baseline_eval:` branch"
    return branches[0]


def _arg_default(func, name):
    args = func.args.args
    defaults = dict(zip([a.arg for a in args[len(args) - len(func.args.defaults):]],
                        func.args.defaults))
    return defaults[name]


def test_every_entrypoint_takes_baseline_eval_defaulting_off():
    _, tree = _src_tree()
    for name in ("_train_impl", "train", "main"):
        default = _arg_default(_func(tree, name), "baseline_eval")
        assert isinstance(default, ast.Constant) and default.value is False, name


def test_baseline_mode_never_resumes_a_checkpoint():
    src, _ = _src_tree()
    assert "resume_dir = None if baseline_eval else latest_checkpoint(ckpt_root)" in src


def test_baseline_branch_evals_and_returns_before_any_publish_or_training():
    _, tree = _src_tree()
    impl = _func(tree, "_train_impl")
    branch = _baseline_branch(impl)
    assert _calls(branch, "_run_heldout_eval"), "the branch must run the held-out eval"
    assert any(isinstance(n, ast.Return) for n in ast.walk(branch)), "the branch must return"
    first_publish = _calls(impl, "publish_policy")[0]
    assert branch.lineno < first_publish.lineno, "a publish would replace the fresh adapter"
    loop = next(n for n in ast.walk(impl) if isinstance(n, ast.For)
                and getattr(n.target, "id", None) == "it")
    assert branch.lineno < loop.lineno


def test_baseline_branch_proves_the_fresh_adapter_is_served():
    _, tree = _src_tree()
    branch = _baseline_branch(_func(tree, "_train_impl"))
    assert _calls(branch, "served_lora_path"), "must check vLLM serves the startup adapter"
    names = {n.id for n in ast.walk(branch) if isinstance(n, ast.Name)}
    assert "adapter_serve_dir" in names


def test_baseline_branch_writes_no_training_state():
    _, tree = _src_tree()
    branch = _baseline_branch(_func(tree, "_train_impl"))
    called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
              for n in ast.walk(branch) if isinstance(n, ast.Call)}
    assert not called & {"write_meta", "save_pretrained", "prune_checkpoints", "MetricsLog"}


# --- per-seed held-out records (both the baseline and every training-time eval) ---------------

def _loop(impl):
    return next(n for n in ast.walk(impl) if isinstance(n, ast.For)
                and getattr(n.target, "id", None) == "it")


def test_heldout_seeds_come_from_one_helper():
    _, tree = _src_tree()
    impl = _func(tree, "_train_impl")
    run_eval = _func(impl, "_run_heldout_eval")
    assert _calls(run_eval, "_heldout_seeds")
    records = _calls(impl, "make_record")
    assert len(records) == 2, "one record in the baseline branch, one in the training loop"
    for call in records:
        assert getattr(call.args[0].func, "id", None) == "_heldout_seeds"


def test_baseline_saves_a_per_seed_record():
    src, tree = _src_tree()
    branch = _baseline_branch(_func(tree, "_train_impl"))
    saves = _calls(branch, "save_record")
    assert len(saves) == 1
    assert "attacker-heldout-baseline" in ast.get_source_segment(src, saves[0].args[0])


def test_training_eval_saves_a_guarded_per_seed_record_per_iteration():
    src, tree = _src_tree()
    loop = _loop(_func(tree, "_train_impl"))
    saves = _calls(loop, "save_record")
    assert len(saves) == 1
    path_src = ast.get_source_segment(src, saves[0].args[0])
    assert "heldout" in path_src and "it" in path_src and "ckpt_root" in path_src
    guarded = [n for n in ast.walk(loop) if isinstance(n, ast.Try)
               and any(getattr(h.type, "id", None) == "Exception" for h in n.handlers)]
    inside = {id(c) for t in guarded for stmt in t.body for c in ast.walk(stmt)}
    assert id(saves[0]) in inside, "a record-writing failure must never stop the run"
