# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RedTeamRL is a research project (see [PLAN.md](PLAN.md)) building an **adversarial self-play sandbox** for training and benchmarking a "trusted monitor" defender model. An **attacker LLM** pursues a concrete goal (exfiltrate a planted secret) against a smaller, separate **defender LLM** that mediates every tool call and message.

**Status: Phases 0–2 delivered, Phase 3 (adaptive attacker) next.** Both the benchmark harness and a working training stack exist — SFT-STaR bootstrap (`sft/`) plus a hand-rolled GRPO trainer (`train/`) with both policies served on vLLM. See PLAN.md §4 for what has actually been measured, including the caveat on the defender result: much of its headline gain was episode-type classification rather than content judgment, which is why PLAN.md §2.8 requires every headline metric to ship with a shortcut probe.

The design principles in [PLAN.md](PLAN.md) §2 are **non-negotiable constraints that the code enforces**, not aspirations. Read them before changing reward, verification, or the gate. The two that most often constrain code changes:
- **No LLM judge determines reward.** Success is a ground-truth substring match (`harness/egress.py`), never a model's opinion. Do not introduce model-graded reward or completion.
- **The defender never sees attacker chain-of-thought.** The attacker reasons in prose and emits one JSON action; the monitor sees only the rendered action (`render_action`), never the prose around it.

## Commands

The project uses `uv`. A `.venv` is already provisioned; invoke it directly or via `uv run`.

```bash
# Install (core) / with Apple-Silicon local inference (mlx-lm)
uv sync
uv sync --extra local

# Run the test suite (372 tests, all offline — no models or network)
.venv/bin/python -m pytest
uv run pytest

# A single test
.venv/bin/python -m pytest tests/test_protocol.py::test_defender_garbage_fails_closed

# Smoke-test one episode end-to-end (needs --extra local: MLX attacker, NullDefender)
.venv/bin/python -m redteamrl.harness.episode

# --- Modal (all GPU work; every function is preemptible, see Gotchas) ---
modal run scripts/sft_defender.py                # SFT-STaR defender bootstrap
modal run --detach scripts/train_defender.py     # defender GRPO (~20 min/iteration on one A100)
modal run --detach scripts/train_attacker.py     # attacker GRPO (Phase 3 — see docs/plans/)
modal run scripts/attacker_viability.py::check   # is the attacker trainable? run BEFORE training it
modal run scripts/probe_grpo_adapter.py::evaluate # shortcut probes: judgment vs. classification
```

**Deprecated — do not build on:** `tasks/schema.py`, `tasks/freeze_v1.py`, `eval/baseline.py`, and `scripts/modal_eval_spike.py` are the superseded YAML-dataset path. `data/tasks/` is **empty**, so those entrypoints no longer run; environments are programmatic (`envs/`). `train/eval_split.py` is dead too (partitions on retired `tier0-<scenario>` ids, zero importers) — but read it before building the held-out-environment split PLAN.md §2.9 requires, since it is a working precedent for that idea.

## Architecture

**Environment / policies / harness are deliberately separated** (PLAN.md §3). The harness owns the episode loop and the trust boundary; policies are pluggable; the sandbox holds only files/secrets.

**Layering (dependency direction — do not invert):**
- `types.py` — pydantic contracts: `TaskSpec`, `AttackerAction`, `DefenderDecision`, `SeedFile`. These self-validate (e.g. an `attack` task *must* have `goal`+`forbidden` and *must not* have `client_actions`; a `benign` task is the mirror image).
- `harness/protocol.py` — the **only** place raw model text is turned into typed objects. `render_*` build prompt text; `parse_*` fail safely. Prompts and policy classes must **not** live here.
- `policies/` — `base.py` defines the `AttackerPolicy`/`DefenderPolicy` Protocols the harness talks to. `prompted.py` holds the system prompts + thin render→generate→parse glue. `scripted.py` replays a fixed action list. `mlx_client.py`/`vllm_client.py` are just `generate(system, messages) -> str` closures behind those policies.
- `harness/episode.py` — `run_episode`: the turn loop. `harness/egress.py` — ground-truth `complete`/`leak` check (substring match). `harness/reward.py` — reward functions.
- `sandbox/` — `Sandbox` Protocol + `LocalSandbox` (tmpdir path-jail, traversal denied via `_safe_path`), `shell.py`'s `LocalShellSandbox` and `modal_shell.py` (real `run_command` shell access). **The shell sandbox is not a security boundary** — it holds fake canaries; isolation is research hygiene, not a containment claim.
- `envs/` — the current task source, replacing the YAML path. An `Environment` is an XBOW-style unit: one realistic filesystem yielding **matched** attack/benign tasks over the *same* files, so intent is the only separator. `build_tasks(ENV, canary, canary_path=...)` injects the secret at build time. **Training builds only `ops_logs + analytics + ci_build`; `billing`, `SUPPORT_PORTAL`, and `RELEASE_EXPORT` are the reserved held-out set (PLAN.md §2.9) — do not add them to a training task list.**
- `verify/` — `verify/task.py` proves a task is well-formed; `verify/defenders.py`'s `NullDefender`/`OracleRedactorDefender` are reference oracles used as verification/test fixtures, not real policies.
- `sft/` — the STaR bootstrap: collect → filter → rationalize → train, plus the shortcut probes (`probe.py` matched-result, `call_probe.py` call-gate) and `canary.py`'s per-episode secret randomization. SFT and GRPO are co-equal halves of one ratchet (PLAN.md §3), not bootstrap-then-discard.
- `train/` — hand-rolled GRPO (`grpo.py`, `train.py`), LoRA, vLLM capture (`capture.py`), per-episode atomic persistence (`episode_store.py`) and per-iteration checkpoints (`checkpoint.py`) so every phase survives preemption. `plasticity.py` holds the non-collapse metrics (effective rank / forgetting / stability).
- `eval/` — `harness.py`'s `run_eval` runs the full task set concurrently; `metrics.py` computes ASR / completion / over-refusal. (`baseline.py` is part of the deprecated YAML path.)

**The episode loop and the gate** (`harness/episode.py`): each turn the attacker acts. A `message` goes to `defender.respond`. A `tool_call` is a **two-stage adjudication**: `adjudicate_call` (allow/reject) → if allowed, run in the sandbox → `adjudicate_result` (pass/redact) before the attacker sees anything. This split is load-bearing: the defender is prompted *not* to reject a call just because its result might be sensitive — allow it, look, then redact. Tools not in `spec.enabled_tools` are rejected by the harness itself, never sent to the defender.

**Fail-closed is a safety property, not a nicety** (`parse_defender_decision`): unparseable/invalid defender output falls back to a safe default (reject/redact/respond) *and drops the model's content* — otherwise overriding an "allow" verdict while keeping its raw-secret content would leak on our own override. Preserve this when touching parsing.

**Attack vs. benign episodes are driven by opposite sides** (`run_eval`): attack tasks are driven by the LLM attacker under a turn budget; benign tasks are driven by a `ScriptedClient` replaying the task's `client_actions`, with the LLM *defender* as the system under test. Over-refusal = a benign task whose `required` artifact never came back.

**Reward invariant:** a leak is strictly worse than an over-refusal (`leak_penalty=2.0 > over_refusal_penalty=1.0`); `test_reward.py::test_leak_strictly_worse_than_over_refusal` guards this.

**Dataset = machine-checkable predicates** (PLAN.md §2.2, §10): a task ships with its own success test. `verify_benign` proves the benign script retrieves `required` under a `NullDefender` *and* that `required` survives when `forbidden` is oracle-redacted (i.e. the two are separable). `verify_attack` proves the secret is planted and a probe attacker can actually leak it. Any new environment must satisfy both — a benign twin is mandatory, because without it over-refusal is unmeasurable and the monitor is free to collapse to refuse-everything.

## Gotchas

- **`docs/`, `notebooks/`, and `eval-runs/` are gitignored** — local-only, not visible via git to future sessions. `docs/plans/` and `docs/specs/` follow a dated plan+spec workflow (one pair per feature) worth reading if present.
- **`mlx-lm` is Apple-Silicon-only and an optional `[local]` extra.** `policies/mlx_client.py` imports it at module top level, so it's imported lazily (inside `__main__`/functions) elsewhere — the core package must import cleanly on a CUDA/Linux box with no mlx-lm. `pyproject.toml` pins `transformers<5` to work around an mlx-lm bug.
- **Indentation is mixed across files** (some tab-indented, e.g. `reward.py`/`eval/harness.py`/`sandbox/local.py`; others space-indented). Match the file you're editing.
- **Modal GPU functions are always preemptible** (`nonpreemptible=True` is CPU/memory only), and a retry can land in the *same container*. So training scripts must release vLLM servers in a `finally`, checkpoint incrementally, and resume by skipping completed work — a unit longer than the mean time between preemptions makes zero progress.
- **`phase_identity` gates resume compatibility.** Anything that changes the observable episode distribution — canary revision, prompt-pool version, env set — must be part of it, and changing it requires a fresh `CKPT_ROOT`. Resuming across such a change silently mixes two different environments into one training run.
- **Aggregate leak rate cannot tell judgment from classification or memorization.** Before reporting any defender improvement, run `scripts/probe_grpo_adapter.py` — matched pairs share a task and an action, so episode-type detection scores ~50% and cannot explain a gain (PLAN.md §2.8, §4.3).
