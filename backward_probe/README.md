# Backward-curriculum diagnostic — Phase 1

**No training, no live GLM calls, no automatic curriculum advancement.** This measures cold
Qwen3-8B's ability to finish real teacher prefixes against the same T2 oracle redactor. It does
not demonstrate learning, fresh-episode success, or superiority of minimal nudges.

Defaults: four distinct training layouts/canaries, depths 1–3, two sampled continuations,
two arms = **48 continuations**. `--samples 1` reduces this to 24. The same source episodes
appear at every depth. Samples/depths are correlated, not independent evaluation worlds.

## Run sequence

CPU preparation imports the immutable source records, selects cases without querying Qwen,
replays them in isolated Modal sandboxes, and caches pinned weights only after a replay succeeds:

```bash
.venv/bin/modal run --detach scripts/backward_curriculum.py \
  --stage prepare --harvest-tag glm53-harvest --tag backward-v1
```

Inspect `backward-curriculum/backward-v1/preparation.json` on `redteamrl-eval-runs`:

```bash
.venv/bin/modal volume get redteamrl-eval-runs \
  backward-curriculum/backward-v1/preparation.json /tmp/backward-preparation.json
```

Then launch the paired diagnostic (one A100-80GB, four concurrent episodes, one-hour timeout):

```bash
.venv/bin/modal run --detach scripts/backward_curriculum.py \
  --stage run --tag backward-v1
```

Download the final result:

```bash
.venv/bin/modal volume get redteamrl-eval-runs \
  backward-curriculum/backward-v1/report.json /tmp/backward-report.json
```

Use the identical command to resume interrupted work. Completed and failed records are preserved;
no automatic retry changes the sampling population. Code, settings, or selected-source changes
require a fresh tag. Never run two writers against the same tag. Preparation and inference are
separate; neither automatically launches training. CPU preparation creates billable small
sandboxes, but does not allocate a GPU or call a teacher.

## What is restored

`start_turn = teacher_length - depth`; `remaining_turns = original_max_turns - start_turn`.
Depth is **not** the student's turn allowance. A teacher that won on turn 7 of 12 leaves six turns
at depth 1. Actual remaining budgets are recorded per case; compare the same worlds across depths.
Maximum depth is configurable through 12, provided all selected sources are long enough. Start
with three to locate a failure boundary, not because three steps would prove learning.

The checkpoint contains the original task/canary (harness-only), prefix actions, attacker-visible
conversation, bookkeeping digest, source provenance and a replay-verified workspace hash. Each
continuation rebuilds the files, re-executes the prefix inside a network-blocked sandbox, checks
saved observations/decisions/execution statuses, checks the workspace hash, and resumes the
existing episode gate. The original remaining budget is preserved, including failed submissions
and invalid actions. Private harness feedback remains visible to Qwen and filtered from defenders.

Legacy harvest records have no live filesystem hashes. Preparation can verify recorded observable
behavior and establish a reproducible workspace hash; it cannot prove equivalence to unrecorded
historical filesystem contents. Hashing covers `/workspace`, not the whole container. Replay
divergence fails closed and is logged. The existing generic restart helper is not used unchanged:
it expects live hashes and restores `_outputs`, not this agent's scratchpad/Tried state.

The same pinned **cold 8B** is used for both arms; the eight-example overfit adapter is not loaded.
It starts with fresh model-authored notes and the original general guidance. Prefix actions and
results remain in Tried. The observation window remains the existing two full results plus Tried
summaries. No new extraction instructions are added to Qwen's system prompt.

- **unassisted:** no teacher notes or rationale. This is still a teacher-generated prefix, not a
  fresh unassisted episode.
- **teacher-pad:** full scratchpad from the saved **input before the restarted decision**, provided
  as fixed read-only reference alongside Qwen's own editable pad. No same-turn completion, chosen
  action, provider reasoning, future observation, or terminal answer metadata is appended. The
  reference remains available during the continuation. No silent tail truncation is permitted.

If the teacher pad contains the canary (literal or one of the supported decoded forms), a witness
in the actual visible observation window is required. Full notes are either retained or rejected,
never quietly rewritten. A grounded literal answer in teacher context is counted separately:
such trials may test copying rather than reconstruction. This screen is not a proof that arbitrary
teacher prose contains no unsupported claims or novel encodings. The full input is saved for audit.

## Budget and failure reporting

Generation uses up to 4,096 tokens, with a 3,072-token thinking budget and 16,384-token context.
Full prompts are checked before every request. The existing agent may make one action-format
repair call per turn; there is no additional missing-reasoning/truncation fallback. All requests,
including repairs and failed generations, retain telemetry. Generation failures are not defender
wins. Sandboxes and the vLLM server are closed in `finally` blocks.

Preparation reports the maximum student turns for the selected cases. Multiply by two for the
maximum generation calls (primary plus repair). The one-hour GPU limit bounds each invocation;
resuming costs additional compute and does not create a cumulative billing cap.

Each per-depth row reports planned/started/completed trials, preparation exclusions, replay skips,
generation failures, wins, token usage, and literal-answer context counts. `win_rate_completed` is
explicitly survivor-only. `win_rate` is unavailable if any started trial failed/skipped. The paired
gap uses only matched completed samples and reports unpaired counts, individual pairs, gains and
losses. Inspect the number of surviving source worlds before interpreting a curve. Full teacher
notes are not a guaranteed ceiling, and their benefit does not predict minimal-nudge benefit.

Artifacts under `backward-curriculum/<tag>/`: `manifest.json`, `selection.jsonl`, `exclusions.json`,
`preflight/`, `cases.jsonl`, `preparation.json`, `continuations/`, and `report.json`. These contain
synthetic canaries for verification; those fields are not serialized into model inputs. All
source worlds must belong to the existing training split. No dev/test world is borrowed to fill
selection, and no source harvest or SFT artifact is modified.

## Key memorization and later work

Inference-only Phase 1 cannot change weights or cause training collapse. Multiple canaries reduce
fixed-answer shortcuts but do **not** make later memorization impossible. Training phases need new
values with faithfully regenerated evidence, split grouping by source/layout, and unassisted
evaluation on different worlds. A successful prefix continuation is not automatically a suitable
SFT example; this package intentionally has no training exporter.

The existing `correction_states` probe used failed-teacher states and one generated action. This
diagnostic uses successful-teacher prefixes and real continuations, so it tests related skills
but is not a directly matched rerun of that 0/4 result.

Phase 2 training, live nudges, hindsight correction, and teacher-free iterative learning remain
separate experiments. Keep a frozen-bootstrap-data control and fresh unassisted evaluation for
the eventual self-improvement claim.

Offline checks:

```bash
.venv/bin/python -m pytest tests/test_backward_probe.py tests/test_restart.py -q
```

Tests use controlled scripted commands only. Real teacher commands are never replayed on the host.
Live sandbox/image reproducibility and GPU behavior still require the two-stage Modal run.
