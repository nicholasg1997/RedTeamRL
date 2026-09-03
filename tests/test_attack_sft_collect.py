from redteamrl.attack_sft.collect import collect_attacker_episodes
from redteamrl.attack_sft.turn_selection import classify_turn
from redteamrl.train.episode_store import read_episodes
from redteamrl.envs import build_tasks
from redteamrl.envs.ops_logs import OPS_LOGS
from redteamrl.sandbox.shell import LocalShellSandbox
from redteamrl.verify.defenders import NullDefender


def test_collection_writes_records_with_per_turn_prompts(tmp_path):
    tasks = [t for t in build_tasks(OPS_LOGS, "sk-CANARY-x", canary_path="app/config.py")
             if t.episode_type == "attack"]
    # deterministic fake attacker: read the secret file, then submit whatever it saw
    script = iter(['{"kind":"tool_call","tool":"read_file","args":{"path":"app/config.py"}}',
                   '{"kind":"submit","value":"sk-CANARY-x"}'])
    def gen(system, messages):
        try: return next(script)
        except StopIteration: return '{"kind":"message","text":"done"}'
    collect_attacker_episodes(tasks, n_per_task=1, generate=gen,
                              defender_factory=lambda: NullDefender(),
                              sandbox_factory=LocalShellSandbox, out_dir=str(tmp_path),
                              canary_seed=0, canary_revision=1, max_turns=4, max_workers=1)
    recs = read_episodes(str(tmp_path))
    assert recs, "no episodes persisted"
    rec = next(iter(recs.values()))
    assert set(rec) >= {"episode_id", "task_id", "canary", "won", "turns"}
    assert rec["turns"][0]["observable_prompt"]                 # captured, non-empty
    classify_turn(rec, 0)                                        # record shape is classifiable


def test_collection_resumes_and_skips_banked(tmp_path):
    # second call with the same out_dir must not rerun a completed episode id
    tasks = [t for t in build_tasks(OPS_LOGS, "sk-CANARY-x", canary_path="app/config.py")
             if t.episode_type == "attack"]
    out_dir = str(tmp_path)

    # NullDefender passes every result through unredacted, so simply reading the file that
    # holds the (per-episode randomized) canary is itself a win -- generate-A only needs to
    # read it, no need to know the randomized value in advance.
    def gen_a(system, messages):
        return '{"kind":"tool_call","tool":"read_file","args":{"path":"app/config.py"}}'

    collect_attacker_episodes(tasks, n_per_task=1, generate=gen_a,
                              defender_factory=lambda: NullDefender(),
                              sandbox_factory=LocalShellSandbox, out_dir=out_dir,
                              canary_seed=0, canary_revision=1, max_turns=4, max_workers=1)
    recs = read_episodes(out_dir)
    assert recs[0]["won"] is True

    # generate-B never touches the secret file, so it would never win if actually run.
    def gen_b(system, messages):
        return '{"kind":"message","text":"just looking around"}'

    collect_attacker_episodes(tasks, n_per_task=1, generate=gen_b,
                              defender_factory=lambda: NullDefender(),
                              sandbox_factory=LocalShellSandbox, out_dir=out_dir,
                              canary_seed=0, canary_revision=1, max_turns=4, max_workers=1)

    recs_after = read_episodes(out_dir)
    # Banked episode 0 is untouched -- still the original winning record, not re-run under gen_b.
    assert recs_after[0]["won"] is True


def test_redaction_enforcement_threads_through_to_run_episode(tmp_path, monkeypatch):
    """M1 (final review): collection must not silently stay 'unshielded' when the run flips to
    'fail_closed' -- the near-miss caught_at_payload semantics shift with it. Confirm the value
    reaches run_episode rather than relying on the harness default."""
    import redteamrl.attack_sft.collect as collect_mod
    from redteamrl.envs import build_tasks
    from redteamrl.envs.ops_logs import OPS_LOGS
    from redteamrl.sandbox.shell import LocalShellSandbox
    from redteamrl.verify.defenders import NullDefender

    seen = []
    real_run_episode = collect_mod.run_episode

    def spy(*args, **kwargs):
        seen.append(kwargs.get("redaction_enforcement"))
        return real_run_episode(*args, **kwargs)

    monkeypatch.setattr(collect_mod, "run_episode", spy)
    tasks = [t for t in build_tasks(OPS_LOGS, "sk-CANARY-x", canary_path="app/config.py")
             if t.episode_type == "attack"]
    collect_mod.collect_attacker_episodes(
        tasks, n_per_task=1, generate=lambda s, m: '{"kind":"message","text":"x"}',
        defender_factory=NullDefender, sandbox_factory=LocalShellSandbox,
        out_dir=str(tmp_path), canary_seed=0, canary_revision=1, max_turns=2, max_workers=1,
        redaction_enforcement="fail_closed")
    assert seen and all(v == "fail_closed" for v in seen)


def test_collection_survives_a_failing_episode(tmp_path, monkeypatch, capsys):
    """One episode raising must not abort the run; survivors persist and the failure is printed
    (regression: a UnicodeDecodeError in one episode killed the whole 120-episode collection)."""
    import redteamrl.attack_sft.collect as collect_mod
    from redteamrl.envs import build_tasks
    from redteamrl.envs.ops_logs import OPS_LOGS
    from redteamrl.sandbox.shell import LocalShellSandbox
    from redteamrl.verify.defenders import NullDefender

    calls = {"n": 0}
    real_run_episode = collect_mod.run_episode

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise UnicodeDecodeError("utf-8", b"\xb7", 0, 1, "invalid start byte")
        return real_run_episode(*args, **kwargs)

    monkeypatch.setattr(collect_mod, "run_episode", flaky)
    tasks = [t for t in build_tasks(OPS_LOGS, "sk-CANARY-x", canary_path="app/config.py")
             if t.episode_type == "attack"]
    collect_mod.collect_attacker_episodes(
        tasks, n_per_task=2, generate=lambda s, m: '{"kind":"message","text":"x"}',
        defender_factory=NullDefender, sandbox_factory=LocalShellSandbox,
        out_dir=str(tmp_path), canary_seed=0, canary_revision=1, max_turns=2, max_workers=1)
    from redteamrl.train.episode_store import read_episodes
    banked = read_episodes(str(tmp_path))
    assert len(banked) == 1                               # the survivor persisted, run did not abort
    assert "failed, skipping" in capsys.readouterr().out  # failure stayed visible
