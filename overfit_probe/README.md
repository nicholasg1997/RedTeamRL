# Small overfit control

Question: **Can a fresh student adapter learn a few existing successful decisions, and does the
saved adapter behave similarly when served through vLLM?** This intentionally evaluates training
states. A pass demonstrates seen-state learning, not generalization, autonomous recovery, or a
stronger defender. A failure at this budget is not proof that the model cannot learn.

## Default experiment

- Reuse `warmstart-v1`: the existing **8B-authored** dataset and its pinned 8B revision. No new
  GLM calls, no re-authoring, no teacher harvest, and no shell/sandbox rollouts.
- Select four pairs from four distinct training layouts: an observed evidence-producing action
  and its later evidence-backed correct submit. Neither dev nor test data is eligible.
- Require a nonliteral evidence witness and its corresponding action in the authored training
  set. Exclude explicit authoring-action references and submits whose answer is already literal
  in the input. These are mechanical checks, not a proof of factual grounding; review the eight
  targets before running.
- Keep the original saved task, scratchpad and observations. The original scratchpad may contain
  the teacher's existing plan. No extra chosen-action hint, reference rationale or target value
  is appended to the evaluation prompt.
- Start from the cold student, with a fresh rank-16 LoRA and learning rate `1e-4`. Use the same
  thinking-on target format and full-completion loss as warm-start. This is a deliberate fitting
  experiment, not a proposed production learning rate.
- Stop at **40 updates or 80,000 supervised tokens**, whichever comes first. Batch four,
  microbatch one. The audited eight 8B targets total **4,169 tokens per pass**, so this bound
  permits roughly nineteen passes. The exact presentation and token counts are reported.
- Greedy generation, thinking on, at most **3,072 generated tokens per decision**, no repair or
  fallback. Use the same limits for cold/tuned HF and cold/tuned vLLM. Generation truncation or
  missing reasoning makes the result inconclusive, not a learning failure.
- Compare eight cold HF decisions, eight tuned HF decisions, then eight cold and eight tuned
  vLLM decisions. The vLLM process is fresh and loads the adapter at startup; no runtime swaps.
- If cold HF already gets at least seven of eight actions and formats right without generation
  errors, skip training and check cold vLLM only. These cases then have too little headroom to
  demonstrate learning under the predeclared criterion, so extra updates would waste budget.
- The GPU function retains the one-hour timeout. Records save after each decision; training
  checkpoints every ten updates and at completion. Exact-command resume preserves completed
  stages and samples. Errors are retained; there are no automatic retries or unbounded loops.

This costs one bounded training experiment and at most 32 single-step generations, rather than
another collection plus full-episode evaluation. Actual runtime still needs a live GPU check.

## Commands

Prepare on CPU first. This verifies the selected examples and caches the pinned model before
allocating an A100:

```bash
.venv/bin/modal run scripts/scratchpad_overfit.py \
  --stage prepare --source-tag warmstart-v1 --tag overfit-8b-v1
```

Inspect the eight selected input/target pairs and the token totals:

```bash
.venv/bin/modal volume get redteamrl-eval-runs overfit-control/overfit-8b-v1/selection.json /tmp/overfit-selection.json
.venv/bin/modal volume get redteamrl-eval-runs overfit-control/overfit-8b-v1/review.jsonl /tmp/overfit-review.jsonl
```

Run the full bounded control:

```bash
.venv/bin/modal run --detach scripts/scratchpad_overfit.py \
  --stage run --tag overfit-8b-v1
```

Rerun that exact command after a timeout. Do not run two copies of the same tag concurrently.
Changing bounds or code requires a fresh probe tag. Existing warm-start source files, datasets,
checkpoints, and reports are not changed. The diagnostic's separate package also keeps it out of
the warm-start and harvest source fingerprints.

To test the existing 14B dataset instead, prepare with `--source-tag warmstart-qwen14b` and a fresh
tag such as `overfit-14b-v1`. The model comes from the source dataset; a tag name does not select a
model. This avoids inadvertently training 14B rationalizations into an 8B model. Selection can
reject a dataset with fewer than four clean pairs; it never fills gaps with held-out examples.

## Reading the result

Download `overfit-control/overfit-8b-v1/report.json`. Other useful artifacts are `fit.json`,
`hf-base/`, `hf-tuned/`, `vllm-base/`, `vllm-tuned/`, and `checkpoints/` under the same directory.

| Verdict | Interpretation |
| --- | --- |
| `PASS_SEEN_STATE_LEARNING` | Both HF and vLLM reach at least 7/8 exact actions and 7/8 correctly formatted responses, with at least two gained cases and no lost cases in each backend; trainable parameters changed. |
| `BASELINE_ALREADY_SOLVES` | Both cold backends already reach 7/8. These states expose a ceiling, not evidence of learning. |
| `INVESTIGATE_SERVING_DIFFERENCE` | HF reaches the action threshold but vLLM does not; investigate loading, numerics and generation behavior before another training run. |
| `FAIL_NO_PARAMETER_CHANGE` | The trainable parameter fingerprint did not change. |
| `INCONCLUSIVE_GENERATION_ERRORS` | A failed/incomplete generation or missing reasoning prevents a clean comparison. |
| `NOT_DEMONSTRATED_AT_THIS_BUDGET` | The clean run did not meet the predeclared learning criterion. Inspect the targets, action losses and decisions; do not call this a general inability to learn. |

Action comparison uses parsed JSON; whitespace is irrelevant. Equivalent alternative commands
are **not** executed or credited. Inspect alternatives before drawing conclusions from an exact
match failure. Submit values must match the verified target exactly for this diagnostic.

Two losses are reported on these same eight **training** cases:

- `all_tokens`: reasoning, notes and action together.
- `conditional_action`: only the JSON action tokens, while supplying the reference rationale and
  notes. This helps detect prose-dominated loss improvements, but remains teacher-forced.

Only the unhinted generation checks measure independent next-action choice. Neither loss is
held-out dev loss, and the seen-state threshold is an engineering criterion, not a statistical
robustness test. No adapter is automatically promoted.

## Where GLM coaching fits

Keep coaching separate from this control. A later small comparison can test the same student with
zero teacher help versus a fixed budget of occasional teacher notes. Preserve the pre-coaching
state, exact inserted note, teacher-call cost and final objective outcome. Guidance must use the
student-visible evidence; supplying the recovered answer tests execution/copying, not discovery.

Assisted wins alone do not demonstrate self-improvement. To test transfer, train on verified
assisted continuations from training worlds and evaluate again without teacher notes on untouched
worlds. No creativity score or model-judged reward is needed for that experiment. The currently
built control makes no teacher calls and does not implement coaching or GRPO.

## Local verification

```bash
.venv/bin/python -m pytest tests/test_overfit_probe.py tests/test_teacher_sft_train.py -q
```

The offline tests cover selection/provenance, no target hints in inference input, action-token
masking, scoring, error denominators, resume, budget bounds and serving aliases/cleanup. Existing
trainer tests cover optimizer updates, unequal-length microbatch weighting and deterministic
resume. Local verification does not substitute for the actual 8B/A100 run.
