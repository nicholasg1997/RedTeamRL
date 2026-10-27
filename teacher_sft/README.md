# Teacher warm-start SFT, first experiment

This pipeline tests whether the 8B can acquire useful recovery behavior from the teacher harvest.
It does not train the defender or port GRPO. A successful T2 run is evidence about a controlled
literal-redaction environment, not a robust defender or a general learning claim.

Everything is isolated in `teacher_sft/` and `scripts/scratchpad_sft.py`. The existing collector and
its source hash are unchanged, so an interrupted `glm53-harvest` can still resume with its original
options. Use the current experimental branch. No old SFT adapters are loaded.

## What is implemented

1. **Import and freeze:** validate the harvest manifest, generated tasks, action/turn alignment,
   submits, and outcomes. Keep raw harvest files immutable. Group by filesystem layout before any
   authoring; assign approximately 80/10/10 train/dev/test. Seeds 8000–8299 assign 244/31/25 episodes
   before accounting for incomplete collection. Derivatives and repeated layouts cannot cross splits.
2. **Select targets:** retain valid, unrepaired decisions from winning episodes, excluding wrong or
   unsupported submits, rejected calls, execution failures, and exact repeated actions. A first
   redacted read can remain useful information gathering. This is turn filtering, not a guarantee
   that every retained action was necessary to win.
3. **Optional correction arm:** at most one candidate per losing episode, after an incorrect submit,
   where a complete real observation is still in the saved prompt and an exact independent decoder
   exposes the correct value. Literal, base64, character-column and byte-hex witnesses are supported.
   Fragment stitching and new exploratory continuations are deliberately not synthesized. These are
   **evidence-checked counterfactual submit targets**, not observed wins or full replay successes.
4. **Student authoring:** the pinned cold Qwen3-8B receives the original prompt plus the selected
   action in a separate authoring suffix. It generates its own reasoning and updated notes. Require
   a complete response, unchanged valid action, notes/action format, and mechanical grounding
   screens. Store raw authoring attempts separately; remove the suffix from training inputs. The
   teacher's private reasoning is not copied. Target-conditioned reasoning is not unaided success.
5. **SFT:** native thinking-on tokenizer format, completion-only loss, no target truncation, fresh
   rank-8 LoRA. Both reasoning and notes/action are supervised; no verbosity or novelty reward.
   The first run stops at **one epoch, 20 updates, or 60,000 supervised tokens**, whichever comes
   first. Whole examples only; the token cap may leave unused budget. Dev loss never updates weights.
6. **Evaluation:** cold versus tuned on the same eight untouched generated worlds plus the registry's
   four reserved attack tasks, with fresh randomized reserved canaries. Identical generation budgets,
   seeds, parser, single-call scratchpad agent and repair policy. Also run up to eight held-out
   wrong-candidate states as a separately reported one-step diagnostic, with no oracle hints in the
   generation prompt. No automatic adapter promotion.

Grounding prose remains imperfect: lexical checks catch some unsupported tokens and hindsight
references, not arbitrary false factual claims. Inspect `review.jsonl` and compare actual behavior.
The correction filter is intentionally narrow; zero eligible correction cases is a valid result.
Failed original continuations are never positive targets.

## Run order on Modal

Run each stage separately. Each remote function retains the **one-hour timeout**. Authoring and
evaluation save individual records; training checkpoints every five updates and at completion.
Rerun the exact command to resume after timeout. Do not run two copies of the same stage/tag at
once. Source or options changes require a new tag; completed authoring rejects are retained rather
than silently retried. Choose the intended bounds before launching.

After the harvest finishes, freeze its completed records on CPU:

```bash
.venv/bin/modal run --detach scripts/scratchpad_sft.py --stage prepare --harvest-tag glm53-harvest --tag glm53-sft-smoke
```

Inspect the summary before any GPU stage:

```bash
.venv/bin/modal volume get redteamrl-eval-runs teacher-sft/glm53-sft-smoke/prepared/summary.json /tmp/glm53-sft-prepared-summary.json
cat /tmp/glm53-sft-prepared-summary.json
```

Missing/exception episodes are explicitly omitted. Preparing a partial harvest freezes only that
snapshot; it does not automatically incorporate subsequently completed episodes. Use a new tag to
prepare a later snapshot. Check there are enough natural train/dev examples and held-out test tasks.

Generate at most 128 student-authored targets (four concurrent requests). The entrypoint first pins
the model revision and downloads it on CPU, then starts the A100:

```bash
.venv/bin/modal run --detach scripts/scratchpad_sft.py --stage author --tag glm53-sft-smoke --max-candidates 128
.venv/bin/modal volume get redteamrl-eval-runs teacher-sft/glm53-sft-smoke/authored/dataset.json /tmp/glm53-sft-dataset.json
.venv/bin/modal volume get redteamrl-eval-runs teacher-sft/glm53-sft-smoke/authored/review.jsonl /tmp/glm53-sft-review.jsonl
```

Read acceptance/rejection counts and the review sample. Stop if accepted reasoning is mostly
fabricated, if format failures dominate, or if natural train/dev examples are absent. Increasing
the budget does not fix bad targets. This bounded authoring sample is an infrastructure/quality
smoke, not the entire harvest. To author more later, choose a new tag and larger bound.

Establish the cold behavioral baseline, then train and evaluate the wins-only arm:

```bash
.venv/bin/modal run --detach scripts/scratchpad_sft.py --stage evaluate --tag glm53-sft-smoke --arm base --n-generated 8
.venv/bin/modal run --detach scripts/scratchpad_sft.py --stage train --tag glm53-sft-smoke --arm wins --max-steps 20
.venv/bin/modal run --detach scripts/scratchpad_sft.py --stage evaluate --tag glm53-sft-smoke --arm wins --n-generated 8
.venv/bin/modal run scripts/scratchpad_sft.py --stage compare --tag glm53-sft-smoke --arm wins
```

Do not launch the next stage until its prerequisite finishes. Training requires the authored set;
adapter evaluation requires the finished training report. The cold baseline runs before training
so it is available even if a training run needs repair. No defender GPU is required for T2.

Only if the first arm remains operational and the correction targets look useful, run the second
arm from the **same cold base**, reusing the authored set and baseline:

```bash
.venv/bin/modal run --detach scripts/scratchpad_sft.py --stage train --tag glm53-sft-smoke --arm 'wins+corrections' --max-steps 20
.venv/bin/modal run --detach scripts/scratchpad_sft.py --stage evaluate --tag glm53-sft-smoke --arm 'wins+corrections' --n-generated 8
.venv/bin/modal run scripts/scratchpad_sft.py --stage compare --tag glm53-sft-smoke --arm 'wins+corrections'
```

Both arms share a token ceiling and update ceiling. Actual tokens can differ because targets stay
whole and dataset sizes differ; reports include exact consumption. Treat this as a small diagnostic
A/B, not a rigorously token-matched comparison if one arm exhausts its data early. Dev loss is
comparable before/after within an arm, not across differing dev mixtures.

## What to inspect

Artifacts are in `redteamrl-eval-runs`, under `teacher-sft/<tag>/`:

| Path | Meaning |
| --- | --- |
| `prepared/summary.json`, `rejections.jsonl` | Counts, omitted episodes and target rejection reasons |
| `prepared/candidates.jsonl` | Selected actions with exact saved prompts and provenance |
| `authored/dataset.json`, `review.jsonl` | Student-authoring yield and review sample |
| `authored/<digest>.json` | Raw authoring prompt, reasoning, response, telemetry, rejection |
| `train-<arm>/encoding.json` | Accepted counts and overlength/format exclusions; never silent truncation |
| `train-<arm>/report.json` | Actual steps/tokens, loss, gradients, adapter path; `promoted=false` |
| `train-<arm>/step-00005/` etc. | PEFT adapter, tokenizer, optimizer/RNG state, completion marker |
| `eval-<arm>/summary.json` | Completed/failed denominators, wins, recovery, primary versus residual invalidity |
| `eval-<arm>/correction-states/` | Held-out one-step correction attempts, separate from episodes |
| `compare-<arm>.json` | Paired gained/lost cases, family metrics and unmatched failures |

Look for finite losses/gradients, nonzero updates, useful fresh recovery, fewer wrong resubmits, and
no large increase in invalid actions, repairs or repetition. Inspect gained **and** lost episodes.
Failure counts are separate from defender decisions. One or two wins in this small evaluation do
not establish improvement; use them to decide whether a larger, frozen evaluation is worthwhile.

## Runtime and testing

CPU model download avoids paying A100 time for the initial transfer. Authoring and evaluation use
four concurrent requests with a bounded context and thinking budget. Training uses bf16, SDPA,
gradient checkpointing, fused AdamW and prompt-logit omission. Only one 8B model is resident per
stage. These choices reduce overhead; actual throughput and memory need a Modal smoke to verify.

Local data preparation is available without Modal:

```bash
.venv/bin/python -m teacher_sft prepare --harvest /path/to/downloaded/harvest --output /tmp/prepared-sft
.venv/bin/python -m pytest tests/test_teacher_sft_data.py tests/test_teacher_sft_train.py tests/test_teacher_sft_eval.py -q
```

Tests use scripted actions and tiny CPU models. They never launch hosted inference or execute
commands taken from imported teacher traces. To additionally test the actual Qwen3 tokenizer and
a tiny Qwen3 model, set `TEACHER_SFT_TEST_TOKENIZER` to an already downloaded tokenizer directory.
GPU image build, 8B authoring acceptance, PEFT training on A100 and real behavioral improvement
remain live-run checks. GRPO's scratchpad/thinking format port remains a separate next project step.
