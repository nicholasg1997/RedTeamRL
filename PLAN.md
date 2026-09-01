# RedTeamRL — Research Plan

**An adversarial self-play sandbox for training and benchmarking a trusted monitor.**

Status: Phases 0–2 delivered; Phase 3 next. Last rewritten 2026-08-08.

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
  (`train/checkpoint.py`) make every phase preemption-tolerant.
- **SFT** — `sft/` (18 modules). Bootstrap + consolidation, not a discardable warm-start; see the
  ratchet below.
- **Non-collapse metrics** — `train/plasticity.py`: effective rank (plasticity), forgetting
  (cycling), stability. Pure functions, offline-tested.

**Deprecated.** `tasks/schema.py` + `tasks/freeze_v1.py` + `data/tasks/` are the superseded YAML
dataset path; environments are programmatic now and `data/tasks/` is empty. CLAUDE.md still documents
them as live and needs the same correction. `train/eval_split.py` is also dead: it partitions on
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

### 4.4 Throughput

~3h → **~20 min per GRPO iteration** on one A100. Root cause of the original figure was a
task-at-a-time rollout loop capping in-flight vLLM requests at ~1, so both engines ran at batch 1;
three separate "GPU contention" symptoms reduced to that one cause. Modal inference baseline
~3005 episodes/hr.

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

### 4.5 Test suite

372 tests, all offline — no models, no network.

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

**Phase 3 — Adaptive attacker, and the environment saturation readout. ← NEXT**
Train the attacker against the frozen trained defender. This is simultaneously a training phase and a
*measurement of the environment suite*: how fast a trained attacker saturates six thin environments is
the quantity that tells us how much authoring Phase 4 needs, and it cannot be obtained any other way.
- Gate first: `modal run scripts/attacker_viability.py::check`. Win rate outside 15–85% means dead
  groups and no gradient; a full run would burn a day producing nothing.
- Align `train_attacker.py` to the opponent config every measured number used (thinking ON, 6144
  tokens) and merge **both** SFT and GRPO adapters into the defender it trains against.
- Offline technique labelling → first diversity graph.
- Carried from Phase 2: re-run the matched probes once the opponent is adaptive. Also the first real
  test of whether `policies/prompt_pool.py` removed the fingerprint or only the one we named — watch
  whether the attacker finds a replacement.
- **Exit:** the attacker measurably raises ASR against the frozen defender; techniques do not collapse
  to a single exploit; we have a saturation curve and a list of what it actually exploited; and the
  probe re-run either moves or indicts the monitor per Phase 2's carried deficiency.

**Phase 4 — Environment authoring framework and scale-up.** The dominant workstream (§5).
- A scenario-authoring abstraction: 2–3 domains expressible in one clean spec, the rest content.
- Per-environment metadata: difficulty, technique tags, canary-placement options.
- Multi-step verifiable predicates; matched benign twins mandatory per §2.2.
- **Held-out environments reserved at the outset per §2.9.**
- Evaluate MORPHEUS as a substrate — its operational verifiers generalize canary-matching to
  programmatic state predicates. Blocking unknown: whether its tool interface can be mediated by our
  monitor. A focused spike answers it.
- **Exit:** a new environment is authorable in a day by following the spec; held-out environments are
  reserved and untouched; the suite is large enough that the Phase 3 attacker no longer saturates it.

**Phase 5 — Population / league self-play.** Both sides co-adapting against pools rather than single
checkpoints (§2.5). Diversity-aware opponent sampling from the offline taxonomy; curriculum driven by
the messy middle; non-collapse measured continuously via `train/plasticity.py`.
- **Exit:** several rounds of monotone-ish improvement on the frozen held-out eval, with forgetting
  and effective rank stable — i.e. improvement that is not cycling.

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
  Mitigations: short horizons first, partial credit, curriculum from the messy middle.
- **Benign-set quality is the sleeper variable.** Whether the monitor learns "allow legit / block
  malicious" or "refuse everything" is mostly determined by the benign half. First-class workstream,
  and per §5 it is half the authoring cost.
- **Known-inert machinery.** The KL anchor is currently negligible (`beta*kl ≈ 6e-5`), and GRPO
  weights all completion tokens equally, putting ~99% of the gradient on prose rather than the verdict
  token. Both are deferred, deliberately, and both should be revisited before Phase 5.
- **Open:** reward weights beyond the leak > over-refusal invariant; turn-budget schedule; how large a
  benign set is enough; whether to migrate off the hand-rolled trainer.

---

## 9. Compute and infra reality

**Modal GPU functions are always preemptible** — `nonpreemptible=True` is CPU/memory only. There is no
stable-GPU escape, so every training phase must be preemption-tolerant: short, incrementally
checkpointed units that resume by skipping completed work. GRPO checkpoints per iteration; SFT
collection commits per episode. A unit longer than the mean time between preemptions makes zero
progress. Retries can land in the *same* container, so servers and GPU memory must be released in a
`finally`, not left to teardown.

Budget degrades gracefully: the benchmark half is inference-only and ships at any funding level; a
single defender or attacker training run is order-\$100s; population self-play across a broad
environment suite is the expensive tier.

---

## 10. Working style

- Build **collaboratively, piece by piece.** Each phase gets its own spec + plan in `docs/` when we
  start it — never a whole-subsystem hand-off.
- Phase gates are real.
- Every task and dataset item ships with a machine-checkable success predicate. That is the spec.
- `docs/` is gitignored on purpose: specs, plans, and `ROADMAP.md` are local-only and never committed.
