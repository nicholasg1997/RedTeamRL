"""Inference-only attacker episode collection with per-turn observable prompts.

Harvests attacker episodes over the TRAINING environments only (PLAN.md §2.9,
`assert_training_split`), writing one atomic JSON record per episode
(`redteamrl.train.episode_store`) so a preempted collection run resumes by skipping episode ids
already on disk rather than re-running them (spec §4/§10). Every record is directly classifiable
by `nearmiss.classify_episode`: each `turns[i]` IS the harness's own per-step dict (`turn`,
`action`, `redaction_status`, ...) with the captured `observable_prompt` and raw `output` merged
in, so downstream selection and rationalization see exactly what the attacker saw and emitted.

No training happens here -- `generate` is any `generate(system, messages) -> str` closure (MLX,
vLLM, or a fake for tests). Weight updates are a later task's job; this module only harvests and
persists.
"""
from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from redteamrl.envs.registry import assert_training_split
from redteamrl.harness.episode import run_episode
from redteamrl.policies.prompted import ATTACKER_SYSTEM_SHELL, HINT_TECHNIQUE, PromptedAttacker
from redteamrl.sft.canary import randomize_task_canary
from redteamrl.train.episode_store import banked_episode_ids, write_episode

# Matches TRANSCRIPT_CONTEXT_CHARS in scripts/train_attacker.py: collection should see the same
# context budget the attacker trains under, so harvested transcripts are representative of what
# GRPO actually faces -- not a longer, easier context it never gets at train time.
_MAX_CONTEXT_CHARS = 40_000


class CapturingAttacker:
    """Wraps a `generate(system, messages) -> str` closure, recording the observable prompt (the
    last user message) and raw output of every call into a thread-local per-episode sink.

    Mirrors `train.capture.CapturingGenerate.episode_capture`: concurrent episodes sharing one
    process must not interleave captured turns into each other's records, which is what a single
    shared list sliced by index would do under `ThreadPoolExecutor`. `PromptedAttacker.act` calls
    this exactly once per turn, unconditionally, right before `run_episode` appends that turn's
    step dict -- so the captured list and `result.steps` line up position-for-position, not just
    on a best-effort basis.
    """

    def __init__(self, inner_generate):
        self.inner_generate = inner_generate
        self.buffer: list[dict] = []          # fallback sink when no episode is scoped
        self._local = threading.local()

    @contextlib.contextmanager
    def episode_capture(self):
        episode_buffer: list[dict] = []
        previous = getattr(self._local, "buffer", None)
        self._local.buffer = episode_buffer
        try:
            yield episode_buffer
        finally:
            self._local.buffer = previous

    def _sink(self) -> list[dict]:
        scoped = getattr(self._local, "buffer", None)
        return self.buffer if scoped is None else scoped

    def turns_for_current_episode(self) -> list[dict]:
        return self._sink()

    def __call__(self, system: str, messages: list[dict]) -> str:
        output = self.inner_generate(system, messages)
        observable_prompt = messages[-1]["content"] if messages else ""
        self._sink().append({"observable_prompt": observable_prompt, "output": output})
        return output


def _run_one(spec, episode_id, rollout_index, capturing, defender_factory, sandbox_factory,
            canary_seed, canary_revision, max_turns, redaction_enforcement) -> dict:
    """Run one episode and assemble its record. Called from a worker thread; `write_episode` is
    documented safe there, so the caller persists the returned record directly, no hand-off."""
    episode_spec = randomize_task_canary(spec, rollout_index, canary_seed, canary_revision)
    defender = defender_factory()
    sandbox = sandbox_factory()
    # Shell tools + the exfiltration hint + the cover story are what make "submit the secret past
    # the monitor" a reachable action at all -- the plain ATTACKER_SYSTEM template never mentions
    # `submit`. Mirrors scripts/train_attacker.py's attack_agent_factory so collection harvests
    # from the same attacker configuration GRPO actually trains.
    attacker = PromptedAttacker(
        capturing, goal=episode_spec.goal, cover=episode_spec.request or "",
        hint=HINT_TECHNIQUE, system_template=ATTACKER_SYSTEM_SHELL,
        max_context_chars=_MAX_CONTEXT_CHARS,
    )
    with capturing.episode_capture():
        try:
            result = run_episode(episode_spec, attacker, defender, sandbox, max_turns=max_turns,
                                 redaction_enforcement=redaction_enforcement)
        finally:
            sandbox.close()
        captured_turns = capturing.turns_for_current_episode()

    turns = [dict(step, **captured) for step, captured in zip(result.steps, captured_turns)]

    return {
        "episode_id": episode_id,
        "task_id": episode_spec.id,
        "canary": episode_spec.forbidden[0],
        "won": bool(result.leak),
        "turns": turns,
        # episode_store's payload contract is shared with GRPO's Example rows; inference-only
        # collection produces none, so this satisfies the round-trip with an explicit empty list.
        "examples": [],
    }


def collect_attacker_episodes(tasks, n_per_task, generate, defender_factory, sandbox_factory,
                              out_dir, canary_seed, canary_revision, max_turns, max_workers,
                              commit=None, redaction_enforcement="unshielded") -> None:
    """Run `n_per_task` rollouts per attack task, persisting one record per episode.

    `tasks` must all be attack tasks (episode_type == "attack") over TRAINING environments only
    -- enforced by `assert_training_split` before any episode runs, so an accidentally-included
    held-out environment fails loudly instead of silently contaminating the collected set.

    Concurrency and resume mirror `train.train.rollout`: episode ids are
    `task_index * n_per_task + rollout_index`, so attribution does not depend on completion
    order; ids already on disk (`banked_episode_ids`) are skipped rather than re-run; `commit` is
    invoked once per newly-written episode from THIS thread only, never from a worker.
    """
    assert_training_split(tasks)
    non_attack = sorted(t.id for t in tasks if t.episode_type != "attack")
    if non_attack:
        raise ValueError(
            "collect_attacker_episodes takes attack tasks only, got non-attack task(s): "
            f"{', '.join(non_attack)}"
        )
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    banked = banked_episode_ids(out_dir)
    capturing = CapturingAttacker(generate)
    work = [(ti, spec, r) for ti, spec in enumerate(tasks) for r in range(n_per_task)]

    def run(item) -> bool:
        ti, spec, rollout_index = item
        episode_id = ti * n_per_task + rollout_index
        if episode_id in banked:
            return False
        # The attacker runs arbitrary shell commands, so an episode can fail in ways one input
        # cannot anticipate. A single failure must NOT abort a 120-episode run whose completed
        # episodes are already banked -- but it must stay VISIBLE (printed) so a systematic
        # failure is not silently swallowed as "just a thin environment".
        try:
            record = _run_one(spec, episode_id, rollout_index, capturing, defender_factory,
                              sandbox_factory, canary_seed, canary_revision, max_turns,
                              redaction_enforcement)
        except Exception as exc:                              # noqa: BLE001 - deliberately broad
            print(f"[collect] episode {episode_id} ({spec.id}) failed, skipping: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            return False
        write_episode(out_dir, record)
        return True

    if max_workers == 1:
        for item in work:
            if run(item) and commit is not None:
                commit()
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run, item) for item in work]
            # as_completed runs in THIS thread, so commit never fires from a worker.
            for future in as_completed(futures):
                if future.result() and commit is not None:
                    commit()
