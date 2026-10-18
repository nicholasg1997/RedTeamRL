# RedTeamRL — Research Plan

**An adversarial self-play sandbox for training and benchmarking a trusted monitor.**

Status: Phases 0–2 delivered; Phase 3 in progress (arm 2 vs arm 3 A/B on attacker lineage v3). Last rewritten 2026-08-08; status sections updated 2026-10-18.

**What this document is.** The stable contract: what we are building, what we will not trade away,
what the code actually does today, and what we have measured. It changes when a *decision* changes,
not when a week's priorities do. Near-term ordering and in-flight bets live in `docs/ROADMAP.md`
(local-only — `docs/` is gitignored and is never committed). If the two disagree, this file is the
one that was reviewed.

We build **collaboratively, piece by piece.** Each phase gets its own spec + plan in `docs/` when we
start it. This document is the map, not the turn-by-turn.

---

## 1. Thesis and deliverables

A tool-mediated sandbox where an **attacker LLM** pursues a concrete goal (exfiltrate a planted
canary) against a smaller, separate **defender LLM** that adjudicates every tool call and every
result before the attacker sees it. Both sides are trainable. Reward is ground truth.

**Two artifacts ship:**

1. **A trained trusted monitor.** A small model that a deployer runs in front of an untrusted agent.
2. **An adaptive-adversary benchmark.** Environments, metrics, and the procedure — reporting attack
   success as a function of *adversary optimization budget* rather than against a frozen prompt set.

**The attacker is co-primary as an engine, not as a release.** It is the automatic curriculum, the
source of adversarial-fine-tuning data, and the thing that makes the benchmark number mean something.
Its *weights* stay internal; what ships is scores, the technique taxonomy, and transcripts.

### Why a monitor rather than a safer model — the open-weights argument

Model-level safety training on open weights is removable by whoever holds the weights: safety
behavior is concentrated in the first few generated tokens and comes off with a small number of
fine-tuning examples. So "a fortified open model" has an incoherent threat model — the attacker can
undo the fortification more cheaply than it was applied.

A monitor inverts that. It is held by the **deployer**, not the attacker, who controls only the input
channel and cannot fine-tune it away. It composes across models, including ones we cannot retrain and
ones released after we trained it. This is why §2.3's smaller/separate constraint is a
deployment-topology requirement, not a budget compromise.

### Why an adaptive adversary rather than a static score

Every published jailbreak benchmark uses a fixed, public attack set, so a number can improve without
robustness changing — the defense only has to cover the set. We measured this pathology inside our
own project (§4.3): a 45-point headline gain that was substantially episode-type classification. An
adversary that retrains against whatever the defense just learned is the only known way to keep the
number from saturating. The output is a **curve, not a scalar**, and we report it as one.

**Obligation this creates.** A benchmark whose adversary is not public is only as trustworthy as the
people running it. We therefore release the environments, the technique taxonomy, sampled attack
transcripts, and the attacker training procedure — enough that a third party can rebuild an
equivalent adversary and check our curve.

---

## 2. Non-negotiables

Constraints, not preferences. Everything else bends to keep these true.

1. **No LLM judge determines reward.** Success is mechanized and ground-truth-verifiable (canary
   substring match, `harness/egress.py`). RL applies sustained optimization pressure and will find any
   judge weakness; "fool the evaluator" is worse than an imperfect static eval.
2. **Benign tasks carry verifiable completion targets.** Every benign task ships a legitimacy context
   and a specific expected artifact, so over-refusal is mechanically detectable. Over-refusal is
   penalized in the same reward as leakage, from the first run, or the defender collapses to
   refuse-everything. A leak is strictly worse than an over-refusal
   (`leak_penalty=2.0 > over_refusal_penalty=1.0`), guarded by
   `test_reward.py::test_leak_strictly_worse_than_over_refusal`.
   **The leak signal is outcome-based and encoding-agnostic.** A leak is a literal secret in what the
   defender passed, or the attacker submitting the literal secret. Encoding/fragment detectors
   (`harness/taint.py::taint_kinds`) are diagnostics only and must never enter the reward — rewarding
   a fixed detector list trains the defender against the list. A leak no detector explains is logged
   as `unattributed` (the signal of a new tactic or a channel around the defender). Tests:
   `tests/test_leak_attribution.py` (including that no training canary appears in attacker-visible
   task text).
3. **Trusted-monitor asymmetry.** The defender is a smaller, separate model. Maps to AI Control's
   trusted-monitor framing, reduces collusion risk, and is what makes the artifact survive open
   weights (§1).
4. **The defender never sees attacker chain-of-thought.** Not at eval, not during training rollouts.
   A monitor trained on the adversary's private reasoning learns mind-reading it will not have in
   deployment. Attacker CoT is logged only for attacker-side training and taxonomy. SFT hindsight is
   the *verifiable outcome*, never an oracle view of intent.
5. **Opponent pools, not just latest checkpoints.** Pools across skill level and technique, to avoid
   narrow overfitting and training-dynamic cycling. **Status: stated, not yet implemented** — the
   defender run to date used a frozen prompted attacker. Built in Phase 5.
6. **Judge LLM is offline categorization only.** Technique labelling and diversity tracking. Never
   part of reward.
7. **CBRN / weapons content is out of scope — hard constraint.** By design, not as a placeholder.
   Agentic harm (exfiltration, unauthorized state change, sabotage) is verifiable in a way content
   harm is not, so breadth never requires a judge back in the loop. See §5 for the carve-outs.
8. **Every headline metric ships with a shortcut probe.** A number is not reportable until paired
   with a measurement that would detect the cheap explanation for it. Aggregate leak rate cannot
   distinguish judgment from memorization from episode-type classification; matched-pair probes can
   (§7). This principle is retroactive — it is what caught §4.3.
9. **Held-out environments are reserved before training begins and are never trained on.** Held-out
   *episodes* test generalization within a world; only held-out *environments* test whether the
   monitor learned transferable judgment. The temptation to defect grows exactly as environments get
   scarce, which is why this is a rule rather than a practice.

---

## 3. Architecture as built

**Environment, policies, and harness are separated; the harness owns the trust boundary.**

```
  attacker (vLLM)  <-->  HARNESS  <-->  defender (vLLM)
                            |
                     shell sandbox (tmpdir)
                   (files / commands / canaries)
```

- **Sandbox** — `sandbox/shell.py` (`LocalShellSandbox`) and `sandbox/modal_shell.py`. A tmpdir with
  real shell access via `run_command`, plus `read_file` / `list_dir`. **Not a security boundary** —
  it holds fake canaries, and isolation is a research-hygiene property, not a containment claim.
- **Environments** — `envs/*.py`. An `Environment` is an XBOW-style unit: one realistic filesystem
  yielding *matched* attack and benign tasks that traverse the same files, so intent is the only
  separator. `build_tasks(ENV, canary, canary_path=...)` injects the secret at build time and expands
  templates into concrete `TaskSpec`s. Six environments exist; three carry the training and probe
  load. **This is the bottleneck — see §5.**
- **Harness** — `harness/episode.py` runs the turn loop. A `message` goes to `defender.respond`. A
  `tool_call` is a two-stage adjudication: `adjudicate_call` (allow/reject) → run in sandbox →
  `adjudicate_result` (pass/redact). The split is load-bearing: the defender is told not to reject a
  call merely because its result *might* be sensitive — allow, look, then redact.
- **Protocol** — `harness/protocol.py` is the only place raw model text becomes typed objects.
  Unparseable defender output **fails closed** and drops the model's content, so overriding an
  "allow" verdict cannot leak via the retained raw text.
- **Reward** — `harness/egress.py` ground-truth canary match; `harness/reward.py` combines leak,
  completion, and over-refusal. Protocol failures cost −2.0, closing a malformed-output reward hack.
- **Training** — hand-rolled GRPO in `train/grpo.py` + `train/train.py`, LoRA adapters, both policies
  served on vLLM on one A100, with runtime LoRA hot-swap between iterations. Per-episode atomic
  persistence (`train/episode_store.py`) and per-iteration checkpoints
  (`train/checkpoint.py`) make every phase preemption-tolerant. Runtime LoRA hot-swap on vLLM 0.21
  has two hard constraints, both found 2026-10-05 and both **silent failures**: (a) the prefix cache
  must be reset after every adapter load (`/reset_prefix_cache`, needs `VLLM_SERVER_DEV_MODE=1`) —
  vLLM 0.21 keys LoRA prefix-cache blocks on the adapter *name*, so earlier swaps reused KV computed
  by the previous adapter; (b) never load with `load_inplace=True` — vLLM keeps the flag and
  re-reads the adapter from disk on every decode step (4–7x slower decode). Both are implemented in
  `policies/vllm_client.py::load_lora_adapter` and pinned by tests; `scripts/vllm_swap_probe.py`
  verifies swaps serve fresh weights. Modal A100 functions now request `cpu=8` (containers
  previously had 1 core).
- **SFT** — `sft/` (18 modules). Bootstrap + consolidation, not a discardable warm-start; see the
  ratchet below.
- **Non-collapse metrics** — `train/plasticity.py`: effective rank (plasticity), forgetting
  (cycling), stability. Pure functions, offline-tested.

**Deprecated.** `tasks/schema.py` + `tasks/freeze_v1.py` + `data/tasks/` are the superseded YAML
dataset path; environments are programmatic now and `data/tasks/` is empty. CLAUDE.md now documents
them as deprecated. `train/eval_split.py` is also dead: it partitions on
`tier0-<scenario>` ids with `HELD_OUT = {"apikey", "smtp"}` and matches nothing in the shell suite.
Worth reading before Phase 4 regardless — it is a **held-out-environment** split, so the mechanism
§2.9 requires already has a Tier-0 precedent to port rather than invent.

### The SFT ↔ GRPO ratchet

SFT and GRPO are co-equal halves of one loop. GRPO *creates* — exploration under verifiable reward
invents behavior nobody demonstrated, with fine per-token credit assignment. SFT *consolidates* —
it takes what was verified-good and makes it reliable, cheap to reproduce, and transferable via
explicit reasoning.

```
GRPO explores + verifies wins on a harder frontier
    -> those verified wins become SFT material
    -> SFT internalizes the reasoning, which transfers to adjacent novel cases
    -> the frontier moves; GRPO explores further
```

SFT draws on two signals. **Signal 1** is per-step mechanical labels, available wherever the correct
action is determinable from observables plus ground truth (a result containing a forbidden substring
⇒ `redact`). Exact, but limited to the locally-visible class. **Signal 2** is outcome-verified
trajectory rationalization: keep trajectories that verifiably won, SFT on the reasoning behind them.
It needs no per-step label and works at any complexity as long as the model *sometimes* succeeds —
which is what GRPO makes increasingly true. The boundary between them is dynamic, which is why SFT
never caps at toy cases.

Signal 2 is noisier (a win may be the opponent's mistake), so three gates hold: a grounding filter
(the reasoning must stand on observables), outcome verification against ground truth, and GRPO's
finer credit assignment as the complement to SFT's trajectory-level filter.

**SFT's job is not perfection.** A defender that is already perfect gives the attacker no gradient.
The target is steady improvement in reasoning quality, with mistakes left for GRPO to fix.

---

## 4. What we have measured

Recorded here so it is not re-derived, and with the caveats attached to the numbers they qualify.

### 4.1 Benchmark baseline (shell tier, prompted policies)

27B AWQ attacker vs. 4B prompted defender: **41.7% attack success, 2.1% over-refusal, 62.5%
benign-leak.** The benign-leak figure is the label-vs-value error — the defender passes secrets
through in *benign* episodes, where the attack framing that would alert it is absent. This is the
messy middle that made the defender trainable at all.

### 4.2 Defender training (SFT-STaR bootstrap + 3 GRPO iterations)

Attack-episode policy leak **78% → 50% → 33%** across iterations (n=18/iteration, monotone,
iter 0 vs. iter 2 **p ≈ 0.007**). Benign leak flat at ~22%; refusal 0% throughout; dead groups
56% → 22%. KL to reference verified at 0.00000 at iteration 0 by merging SFT into the backbone and
training a fresh adapter on top.

**Caveat (found 2026-10-05): the iteration ≥1 numbers above were generated by a stale-KV hybrid
policy.** vLLM 0.21 keys LoRA prefix-cache blocks on the adapter *name*, so after each swap the
defender system prompt's KV was still the iteration-0 adapter's; the new weights only acted on
uncached tokens, and the iteration-1 update trained on those samples as if on-policy. The 50% / 33%
points are therefore not clean evidence of per-iteration learning (iteration 0 is clean). Unaffected:
§4.3 and §4.6, which merge the `iter1` checkpoint into a fresh server with no swap — they validly
measure what those weights do. Fixed in `load_lora_adapter` (prefix-cache reset after every load,
verified by `scripts/vllm_swap_probe.py`); re-run iterations 0–2 to restore the trend claim.

### 4.3 What that result actually was — the finding that shaped §2.8

Deterministic matched probes, where the two halves share a task and an action and differ only in
whether the credential field holds a usable secret:

| probe | sft | sft+grpo |
|---|---|---|
| `pair_taint_accuracy` (classifier test) | 12.5% | 12.5% |
| `call.reject_recall` (the call gate) | 0% | 0% |
| `taint_cleared_given_attempt` (unseen secrets) | 21.1% | 36.8% |

A pure episode-type classifier scores ~50% on `pair_taint_accuracy` regardless of how good its type
detection is, because it must get both halves right; the probe's secrets are derived independently of
training canaries, so memorization gains nothing either. Both hard zeros are unmoved.

**Reading:** much of the 45-point aggregate gain came from classifying episode type against a frozen
opponent, not from judging content. `taint_cleared_given_attempt` 21% → 37% is genuine transfer.
Both are true.

Two limits on that positive result. The probe suite is coarse (48 cases) — raise
`COUNTERFACTUAL_PROBE_VARIANTS` before leaning on small movements. And `probe_grpo_adapter.py` builds
**only `ci_build`, a trained-on environment**, so the transfer demonstrated is to unseen *secrets
within a seen world*, not across environments. Re-pointing the probe at a reserved environment (§5)
upgrades the claim and costs nothing.

This is a fact about the *environment*, not a failure of the method: a frozen, single-prompt opponent
leaves a fingerprint lying around, and RL takes it. The two fixes are structural (style-paired prompt
pools, `policies/prompt_pool.py`, which deny the fingerprint we named) and adversarial (a trained
attacker, which denies the ones we did not).

**Credit structure (offline analysis of the r9 defender rollouts).** In leak episodes roughly one
decision in 5–9 is at fault, so episode-level GRPO assigns the same negative advantage to the
~80–96% of decisions that were correct. The deciding decision is identifiable from ground truth (the
result exposing the secret), which makes defender per-step credit cheap.

### 4.4 Throughput

The original ~3h → ~20 min per GRPO iteration gain came from fixing a task-at-a-time rollout loop that
capped in-flight vLLM requests at ~1, so both engines ran at batch 1; three separate "GPU contention"
symptoms reduced to that one cause. Modal inference baseline ~3005 episodes/hr.

**The "~20 min per iteration" figure was measured with both swap bugs in §3 (stale prefix KV,
`load_inplace` re-read) plus a 1-CPU container, and is superseded.** Measured 2026-10-06 on one
A100-80GB with both models co-located: attacker iteration-0 rollout of 96 episodes **28 min → ~4
min** (attacker decode ~3 → ~22 tok/s per sequence, matching an isolated benchmark);
`ROLLOUT_WORKERS` / `max_num_seqs` raised 16 → 48. The GRPO update (previously ~22 min, one example
per forward pass) is now batched (`Learner.logprobs_batch` + `train.micro_batches`, proven
math-identical by `tests/test_batched_update.py`) — and measured **no speedup**: ~21 min for 536 live
examples (v3 arm 2 iter 0) vs ~22 min for 627 before. At ~2k tokens per example a single forward is
already compute-bound (~18 forward-equivalents x 536 x 2k tokens x ~16 GFLOP/token ≈ 3e17 FLOP ≈ 25
min), so batching cannot help. The update is now ~80% of an iteration; real levers cut FLOPs:
fewer inner epochs (ratio reached only 1.0003 — clipping never fires), skipping checkpoint
recompute where memory allows, and shared-prefix scoring (consecutive prompts share ~96% of tokens). A split-defender option (`--split-defender`, defender on its own L4) exists but is
unnecessary after the fixes. Diagnostic tooling: per-process `[cpu]` logging
(`train/cpu_monitor.py`) and `scripts/vllm_decode_bench.py`. Lesson: aggregate CPU/GPU metrics hid a
single saturated thread; stack sampling (py-spy) found it.

### 4.6 Attacker viability gate (2026-08-31) — Phase 3 precondition

Qwen3-8B attacker vs. the full trained defender (base + SFT + GRPO), 48 episodes per config:

| config | win rate | invalid actions | verdict |
|---|---|---|---|
| thinking-6144 | 18.8% (9/48) | **29.5%** | TRAINABLE |
| no-thinking-2048 | 16.7% (8/48) | **0.4%** | TRAINABLE |

**The gate passes.** Win rates are one episode apart — at n=48, p≈0.17 the standard error is ~5.4%,
so the configs are indistinguishable on capability. What separates them is truncation: a generation
that hits the token cap never closes its `<think>` block and emits no parseable action. Because a
truncated generation is a MAX-LENGTH sequence and GRPO weights all completion tokens equally,
roughly 60% of the gradient at thinking-6144 would come from generations that produced no action.
Training therefore runs thinking-OFF at 2048.

This corrects an assumption made when the phase was planned: that the attacker had to match the
sampling config behind the defender's benchmark numbers. That opponent was a 27B AWQ, not an 8B —
the model already differed, which is the variable that dominates, and matching sampling bought
nothing measurable.

**Position in the band.** 17% sits at the low end of the 15–85% trainable range, predicting ~23%
dead groups at the training group size of 8. Group size 8 is the right choice: at this win rate it
yields the most live groups per episode spent (9.6 per 100, versus 7.4 at G=12 and 5.9 at G=16 —
larger groups have diminishing returns). If the attacker stalls in early iterations, the lever is a
weaker opponent — drop `DEFENDER_GRPO_ADAPTER` to `None` for an SFT-only defender — not a bigger
attacker.

**Caveat (2026-10): the 16.7% does not describe current training.** It was measured on the older thin
envs at canary revision 101, without the attacker's output history, with the SFT-era config. Current
training (procedurally generated envs, rev 102, raw base attacker, history K=3) starts at **~9% win
rate** at iteration 0. `ACKNOWLEDGED_CONFIG_DELTAS` in `scripts/train_attacker.py` records that a
re-gate on the current config is owed. The dead-group expectation above is correspondingly stale: at
~9% win rate and group size 8, ~47% dead groups is expected (0.91^8); measured 42–50%.

### 4.7 Attacker SFT collection readout (2026-09-02) — the failure distribution is learnable

First `attack_sft` collection, base 8B attacker vs. the full trained defender, 120 episodes over the
three training envs: **`win=13  near_miss=107  other=0`.**

- **`other=0`**: the attacker never refuses and never fails to engage — every episode reaches the
  secret. (This settles the open "is the attacker refusing?" question: no.)
- **`win=13`** (~11%): consistent with the GRPO base rate, so it is stable.
- **`near_miss=107`** (89%): the attacker reaches the payload almost every time and loses at the *last
  step* — a wrong submit (fumbled finish) or a payload the defender redacted (caught at payload). This
  is the most learnable failure distribution available: the pivotal moment is anchored in every case,
  and it clears `MIN_EXAMPLES` comfortably, so SFT proceeds.

**What it says about the environments:** exploration is trivial (the secret is always found), so the
entire game lives at exfiltration against an actively-redacting monitor. That is a real, learnable
game — enough to prove the machinery here — but a *narrow slice* of the eventual threat model, which
is the Phase-4 argument (§5). **Two next-iteration refinements it surfaced:** the readout should split
fumbled-finish vs. caught-at-payload (attacker's own error vs. the defender catching it — different
lessons), and `NEARMISS_FRAMING` currently talks about evading detection, which fits caught-at-payload
but mis-frames a fumbled finish (an accuracy error, not a detection one).

**Follow-up.** The reflective attacker SFT that followed collapsed (held-out win rate ~21% → 2.3%;
it memorized a single-file loop). It was redesigned (masking, short directive reasoning) and then
re-scoped: SFT is no longer a pre-RL bootstrap for the attacker. Current order is **RL first**; SFT
returns afterwards as a legibility-distillation pass over the model's own reasoning from winning
rollouts, with a faithfulness check (spec deferred).

### 4.5 Test suite

709 tests, all offline — no models, no network.

---

## 5. The bottleneck: environments

Everything downstream is gated on this, and it is the dominant workstream from Phase 4 onward.

Today's environments are **single-file and roughly one step**. The thesis needs realistic, multi-step,
richly-tooled scenarios — billing, HR, cloud infra, git repos, CI/CD, healthcare, finance — supporting
chains like *list dir → read docs → infer where credentials live → use prior information → extract*.
`run_command` already provides rich tool use; **what is missing is authoring, not plumbing.** The
required lift is a scenario-authoring abstraction: plausible file trees plus documentation plus
credentials planted where they would realistically live, with verifiable multi-step predicates.

**Why this is not just a content backlog.** With six environments, a trained attacker finds the two or
three exploits that exist and stops — collapse by construction — and any benchmark number generalizes
no further than the environment distribution. Without breadth we cannot distinguish a monitor that
learned judgment from one that overfit six worlds.

**Reference format — XBOW `validation-benchmarks`.** 104 self-contained benchmarks, each a directory
with a Docker service definition and a `benchmark-config.json` carrying name, description, difficulty
(1–3), tags, and a `win_condition` of `flag` or `question`. Critically, **the flag is injected as a
build argument rather than hardcoded** — the same design as our build-time canary injection, which our
per-episode canary randomization extends.

**Where we must diverge.** XBOW benchmarks are offense-only: did the attacker find the flag. A monitor
benchmark needs the **benign twin** — legitimate tasks over the same files with verifiable completion
artifacts — or over-refusal is unmeasurable and the defender is free to collapse to refuse-everything.
That roughly doubles authoring cost per environment and is the real reason this is the bottleneck.

### The held-out set already exists — it was never used as one

Every training and SFT script builds exactly `ops_logs + analytics + ci_build`. **`billing`,
`SUPPORT_PORTAL`, and `RELEASE_EXPORT` have never been trained on by anything.** §2.9 is therefore
not blocked on scarcity; the reserve exists and has simply never been evaluated against. Two
consequences: the strongest available generalization measurement is currently free, and the one
positive result we have (§4.3) is weaker than it looked, because the probe runs on a trained-on world.

**Build the real mechanism rather than porting the Tier-0 one.** `train/eval_split.py` proves the idea
but is a hardcoded pair of names with no enforcement — adapting it just relocates the hardcoding. What
§2.9 actually needs:

- **Membership declared in environment metadata**, not in a split module, so adding an environment
  forces the train/held-out decision at authoring time instead of deferring it.
- **Stratified reserves** — held out by domain *and* technique, so the reserve is not accidentally
  all-easy or all-one-mechanism. With only three reserved this is a judgment call; past ~20
  environments it must be systematic.
- **Enforcement by test.** A test that fails if a reserved environment id appears in any training task
  list. Without it §2.9 is an intention, and intentions lose to a scarce-environment deadline. This is
  the part with no precedent, and it is the part that makes the rule real.
- **Applies to attacker training too**, not just the defender. An attacker that has seen every
  environment cannot measure cross-environment transfer either.

**The second bottleneck — the adversarial population loop** — is a strong attacker as automatic
curriculum (the UED/PAIRED/league insight): the difficulty signal is the *messy middle*, near-50%
win rate = maximal regret = maximal learning signal. Current read: this is tractable once the
training loop is correct, and is sequenced after environments rather than before.

---

## 6. Phases

Gates are real. We do not start Phase N+1 until Phase N's exit criteria are met.

**Phase 0 — Toy GRPO. ✅** LLM-RL mechanics de-risked on a verifiable toy task.

**Phase 1 — Sandbox + benchmark harness. ✅** Delivered as programmatic shell environments with
matched attack/benign pairs rather than the originally-planned frozen YAML dataset. Concurrent eval
harness, ASR / completion / over-refusal metrics.

**Phase 2 — Defender vs. frozen attacker. ⚠️ Passed with a carried deficiency.** SFT-STaR bootstrap +
GRPO; result and caveat in §4.2–4.3. The exit criterion "beats the baseline on held-out attacks" is
met on aggregate leak and **not** met on the matched probes. Treat the trained monitor as a working
artifact, not a validated one.

*Why the gate opens anyway, given that gates are real.* The probe failure has a diagnosed cause —
a frozen single-prompt opponent leaves an episode-type fingerprint, and RL takes it — and re-running
Phase 2 against that same opponent would reproduce the same result rather than fix it. Phase 3 *is*
the intervention. The deficiency is therefore carried forward as **Phase 3's entry condition**: the
probes are re-run against the adaptive attacker, and if `pair_taint_accuracy` and `reject_recall`
are still pinned once the opponent moves, the problem is the monitor and not the environment, and
Phase 2 reopens. This is the only phase permitted to advance on a diagnosed deficiency, and only
because the next phase is the diagnosis's remedy.

**Phase 3 — Adaptive attacker (prove the ratchet works on the attacker side). ← IN PROGRESS**
Train the attacker against the frozen trained defender. **The goal is to prove the SFT→GRPO
*machinery* works on the attacker side too (it is only defender-proven) and yields a modest lift —
NOT to produce a strong exploiter.** The attacker's ceiling is bounded by environment richness, so a
low absolute win rate on the current thin envs is a fact about the environments, not a method failure
(see §5); do not chase strong exploitation here — that needs Phase 4's environments.
- ✅ Viability gate passed (§4.6): 8B trainable at ~17%, thinking OFF / 2048 tokens (thinking-6144
  truncated 29.5% of actions for no win-rate gain — do NOT use it). Re-gate on the current config is
  owed (§4.6 caveat).
- Attacker SFT attempted (`attack_sft/`, §4.7), collapsed, redesigned, then re-scoped to a post-RL
  legibility-distillation pass.
- **Current experiment: arm 2 vs arm 3** (gating a tree-based Monte-Carlo GRPO). Arm 2 = plain
  episode-level GRPO (the baseline every later method is judged against). Arm 3 = arm 2 + a
  deterministic exact-repeat penalty (coef 0.05; still ground-truth, no judge). Compared at **matched
  generated tokens** (`cum_gen`), trailing-window means, with a pre-registered kill criterion: shelve
  the tree iff arm 3's stalled rate ≤ arm 2's and arm-3 win rate exceeds arm 2's by ≥3pp with
  bootstrap-CI lower bound > 0 (details in the local spec
  `docs/superpowers/specs/2026-10-04-tree-mc-grpo-design.md` §13).
- **Attacker lineage v3** (fresh checkpoint roots `attacker-grpo-v3-*`): mean-centred advantages
  (r − mean, no std division — std normalisation amplified one-repeat differences into full-strength
  gradients; Dr. GRPO), LR 2e-5 to compensate, batched update, and the attacker sees its own last 3
  outputs in its own prompt (never the defender's view). Earlier lineages (v1 arm-2 iterations, the
  "stalled at 12%" first pass) are superseded; the 12% result was likely contaminated by the
  prefix-cache bug (§3).
- Tree-MC-GRPO design (local spec): forkable episode state, per-edge credit Q(child) − V(parent),
  driver pruning with no prune penalty, restart-from-state buffer, MAX_TURNS curriculum (12 → 20 → 30
  keyed to win rate vs the current defender).
- Offline technique labelling → first diversity graph.
- Carried from Phase 2: re-run the matched probes once the opponent is adaptive — the first real test
  of whether `policies/prompt_pool.py` removed the fingerprint or only the one we named (watch whether
  the attacker finds a replacement).
- **Exit (revised):** RL measurably moves the attacker on v3 (win rate up and dead groups down over a
  trailing window at matched tokens), the arm 2 vs arm 3 result decides whether the tree is built
  now, and the carried probe re-run is done. The probe re-run either moves or indicts the monitor per
  Phase 2's carried deficiency.

**Phase 4 — Environment authoring framework and scale-up.** The dominant workstream (§5).
- **The reusable lever is the *framework*, not hand-built one-offs**: a scenario-authoring abstraction
  (file tree + planted secret + verifiable multi-step predicate + mandatory benign twin) so envs
  become cheap content. Build this regardless of the 3↔4 ordering.
- **The Phase 3 → Phase 4 order is data-dependent, arbitrated by the collection readout.** A working
  attacker is the instrument that surfaces what to build (PLAN §5, "attacker as diagnostic"), so the
  default is attacker-first. But if the readout says the envs are too thin to even demonstrate the
  machinery, environments jump ahead of *finishing* attacker training. Do not hard-commit the order.
- Per-environment metadata: difficulty, technique tags, canary-placement options.
- Multi-step verifiable predicates; matched benign twins mandatory per §2.2. Realistic depth so
  exploration is a real challenge — the current envs make finding the secret trivial and put the whole
  game at exfiltration (§4.7), which is a narrow slice of the eventual threat model.
- **Held-out environments reserved at the outset per §2.9.**
- Evaluate MORPHEUS as a substrate — its operational verifiers generalize canary-matching to
  programmatic state predicates. Blocking unknown: whether its tool interface can be mediated by our
  monitor. A focused spike answers it.
- **Exit:** a new environment is authorable in a day by following the spec; held-out environments are
  reserved and untouched; the suite is large enough that the Phase 3 attacker no longer saturates it.

**Phase 5 — Alternating self-play → population / league.** Freeze one side, train the other to
saturation (measured on *held-out*, not training, performance), swap, repeat several rounds. Two
requirements make "several rounds" converge instead of oscillate:
- **Opponent pools, not just the latest checkpoint** (§2.5). Keep the best checkpoint of each round and
  sample opponents from the pool. Without this the defender beats the *latest* attacker while
  forgetting how to beat an older one, and the pair cycles forever. This is the AlphaZero-league fix;
  it is the mechanism, not a nicety. Pools keep the best attacker and defender **per tactic type**
  (niches from the `leak_attribution` diagnostic; `unattributed` gets its own slot), and **every new
  defender checkpoint is re-measured on the benign twins (over-refusal) before entering the pool** —
  a defender that improves by refusing more is the classic collapse.
- **Benchmarking is a per-round GATE, not a final phase** (see §7). Every alternation emits held-out
  environment transfer, the matched probes (judgment vs. classification), and *attacker-vs-the-base-
  defender* (absolute improvement, not just relative to the current opponent). Non-collapse
  (forgetting / effective rank) measured continuously via `train/plasticity.py`. If measurement waits
  until the end, you cannot tell which round was real improvement versus cycling versus overfitting the
  current opponent — and it is unrecoverable after the fact.
- Diversity-aware opponent sampling from the offline taxonomy; curriculum driven by the messy middle.
- **Exit:** several rounds of monotone-ish improvement on the frozen held-out eval, with forgetting and
  effective rank stable — i.e. improvement that is not cycling.

**Phase 6 — Benchmark capstone and monitor release.** Snapshot a versioned suite; report the
adversary-budget curve; publish monitor weights, environments, taxonomy, transcripts, and procedure.
- **Exit:** held-out-adversary and held-out-environment numbers hold up, and an external benchmark
  (AgentDojo / InjecAgent) corroborates.

---

## 7. Evaluation

**The held-out ladder.** Each rung tests something the rung below cannot.

| rung | tests | status |
|---|---|---|
| Held-out **episodes** | generalization within an environment | ✅ `sft/review_split.py` |
| Held-out **canaries** | memorization of specific secrets | ✅ per-episode randomization |
| Matched-pair **probes** | judgment vs. episode-type classification | ✅ `sft/probe.py`, `sft/call_probe.py` |
| Held-out **environments** | transferable judgment vs. overfitting the training worlds | ⚠️ reserved, never used — §5 |
| Held-out **adversary** | robustness vs. beating one opponent | ⬜ Phase 5 |
| **External** benchmarks | our framing vs. the field's | ⬜ AgentDojo / InjecAgent, never wired |

Everything above the probe rung is a *training* measurement. Only the bottom three produce a
**benchmark** number.

**Benchmarking steers; it does not just validate — so it runs every round, not at the end.** An
adaptive-adversary project only works if measurement decides what to build next, because a static
number saturates (§1). The held-out ladder, the matched probes, and the pool checks are therefore a
**per-round gate on the alternating loop (Phase 5)**, not a final validation phase. Run them once at
the end and you cannot distinguish real improvement from cycling from overfitting the current opponent
— and it is unrecoverable after the fact. "Saturation" is defined on held-out, not training,
performance. This is the single most consequential lesson of the Phase-2 defender result (§4.3): the
probe is what caught the metric being gamed, and it only helps if it fires while there is still a
decision to make.

**Core metrics.** Attack success rate (↓), benign completion rate, over-refusal rate — measured on the
frozen held-out eval on a fixed cadence, because internal win rates cannot distinguish improvement
from cycling.

**Shortcut metrics (§2.8).** `pair_taint_accuracy` and `pair_exact_accuracy` (classifier test —
matched halves, so type detection cannot help), `verb_pair_accuracy` (the "reject the scary verb"
test — `od`, `base64`, `cut` appear on both sides), `reject_recall` and `allow_precision` on the call
gate.

**Non-collapse metrics.** Effective rank (plasticity), forgetting (per-task regression from its own
peak = cycling), stability (within-task reward variance, read *with* the mean). Adaptation speed and
recovery time need a defined regime shift and apply only once the opponent population shifts.

**Headline claim, falsifiable.** The trained monitor reduces held-out ASR by X% while holding benign
completion ≥ Y%, *at a stated adversary optimization budget*, on **held-out environments**. X and Y
are set before Phase 5 runs.

---

## 8. Risks and open questions

- **Environment breadth is the binding constraint** (§5). Every other risk is downstream.
- **Shortcut learning is the default, not the exception.** Measured in §4.3. Any frozen component —
  opponent, prompt, canary, environment — becomes a fingerprint that RL will take instead of doing the
  work. Mitigations: prompt pools, adaptive attacker, canary randomization, held-out environments.
- **Degenerate self-play equilibria.** Pools + diversity sampling + plasticity metrics (Phase 5).
- **Sparse long-horizon credit assignment.** Single-bit leak/no-leak over a multi-turn rollout.
  Mitigations: short horizons first, partial credit, curriculum from the messy middle. The planned
  mitigation is the tree design (per-step credit, restart-from-state; §6 Phase 3).
- **Benign-set quality is the sleeper variable.** Whether the monitor learns "allow legit / block
  malicious" or "refuse everything" is mostly determined by the benign half. First-class workstream,
  and per §5 it is half the authoring cost.
- **Known-inert machinery.** The KL anchor is currently negligible (`beta*kl ≈ 6e-5`), and GRPO
  weights all completion tokens equally, putting ~99% of the gradient on prose rather than the verdict
  token. Both are deferred, deliberately, and both should be revisited before Phase 5.
- **Open:** reward weights beyond the leak > over-refusal invariant; turn-budget schedule (draft:
  MAX_TURNS curriculum in the tree spec); how large a benign set is enough; whether to migrate off
  the hand-rolled trainer.

---

## 9. Compute and infra reality

**Modal GPU functions are always preemptible** — `nonpreemptible=True` is CPU/memory only. There is no
stable-GPU escape, so every training phase must be preemption-tolerant: short, incrementally
checkpointed units that resume by skipping completed work. GRPO checkpoints per iteration; SFT
collection commits per episode. A unit longer than the mean time between preemptions makes zero
progress. Retries can land in the *same* container, so servers and GPU memory must be released in a
`finally`, not left to teardown.

Budget degrades gracefully: the benchmark half is inference-only and ships at any funding level; after
the 2026-10 throughput fixes an attacker GRPO iteration costs roughly $0.50–1 on one A100-80GB
(rollout ~4 min; update under measurement); `--until-iter N` and `--skip-final-update` bound a launch
for budget-capped runs and resume without lost work; population self-play across a broad environment
suite is the expensive tier.

---

## 10. Working style

- Build **collaboratively, piece by piece.** Each phase gets its own spec + plan in `docs/` when we
  start it — never a whole-subsystem hand-off.
- Phase gates are real.
- Every task and dataset item ships with a machine-checkable success predicate. That is the spec.
- `docs/` is gitignored on purpose: specs, plans, and `ROADMAP.md` are local-only and never committed.
