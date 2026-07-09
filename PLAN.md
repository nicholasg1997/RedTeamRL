# RedTeamRL — Research Plan

**An adversarial self-play sandbox for training and benchmarking a trusted defender model.**

Status: planning. This document is the master phased plan. Each phase gets its own
detailed implementation plan (written when we start that phase). We build this
**collaboratively, piece by piece** — not as a single hand-off.

---

## 1. Thesis / contribution

A Docker-isolated, tool-mediated environment where an **attacker LLM** pursues a concrete
malicious goal (exfiltrate a credential, leak PII, extract a system prompt, escalate a
tool permission) against a smaller, separate **defender LLM** that mediates every tool
call and request. The environment yields two artifacts:

1. **A standalone agentic-safety benchmark** (inference-only, no training required).
2. **A trained "trusted monitor" defender**, improved via self-play RL against an evolving,
   diverse pool of attackers.

**What's novel is the combination:** ground-truth *verifiable* reward with **no LLM judge in
the loop** + a **trusted-monitor asymmetry** (defender is smaller/separate) + **over-refusal
penalized in the same reward** as leakage + **technique-diversity opponent pools**.

---

## 2. Core design principles (the non-negotiables)

These are constraints, not preferences. Everything else bends to keep these true.

1. **No LLM judge determines reward.** Success is mechanized and ground-truth-verifiable
   (canary-token / secret matching). Rationale: RL applies sustained optimization pressure
   and will find any judge weakness — "fool the evaluator" is a worse problem than an
   imperfect static eval.
2. **Benign tasks carry verifiable completion targets.** Every benign task ships with
   (a) a legitimacy credential/context justifying the request and (b) a specific expected
   output artifact the defender must produce. Over-refusal is then mechanically detectable
   (the artifact was not produced). Over-refusal is penalized **as strongly as** leakage,
   from the first training run — otherwise the defender collapses to refuse-everything.
3. **Trusted-monitor asymmetry.** The defender is a *smaller, separate model* from the
   attacker — different weights, different capability tier. Maps to AI Control's trusted-
   monitor framing and reduces collusion risk.
4. **Attacker CoT is a training-time teacher signal only — never a live defender input.**
   The defender's policy input must never contain the attacker's chain-of-thought, not even
   during training rollouts (or it learns to condition on info it won't have at deployment).
   Attacker CoT enters only via the offline CoT-distillation stage, as a teacher signal that
   shapes the training target. Eval always hides attacker CoT. This is asymmetric
   actor-critic: the teacher sees privileged state; the deployed defender never does.
5. **Opponent pools, not just latest checkpoints.** Maintain pools of attacker and defender
   checkpoints across skill levels and technique types to avoid narrow overfitting and
   training-dynamic cycling.
6. **Judge LLM is offline categorization only.** Used to label attack technique and track
   diversity over time. Never part of the reward signal.
7. **CBRN / weapons content is out of scope — hard constraint.** Given the dual-use risk of
   training and potentially open-sourcing an attacker, this category is excluded by design,
   not as a placeholder.

---

## 3. Architecture

**Separate the environment from the policies.**

- **Environment (sandbox container):** holds only the tools, files, and canary secrets — the
  "world." Reset to a template between episodes (cheap: tmpfs/overlay snapshot). No model
  weights inside. During training the secrets are fake canaries, so heavy isolation is not
  required here; strong isolation (gVisor) is reserved for any released attacker artifact.
- **Policies (attacker, defender):** run as **inference servers (vLLM) outside** the
  container. Many environments can share one served model.
- **Harness (orchestrator):** runs the episode loop. Routes attacker turns → defender →
  tool calls into the sandbox → results back. Enforces the trust boundary: every attacker
  tool call and request passes through the defender first. Computes the mechanized reward
  from ground-truth (did the canary leak? was the benign artifact produced?).

```
  attacker (vLLM)  <-->  HARNESS  <-->  defender (vLLM)
                            |
                     sandbox container
                   (tools / files / canaries)
```

**Episode reset, not respawn:** keep the sandbox container warm; reset its filesystem/state
to a per-task template each episode. Point the two inference endpoints at it. This is what
keeps rollout throughput viable at RL scale.

**Defender action space (Phase 1 decision):** the defender's core action is the *gate
decision + response*, with read-only investigation tools. Keep write/exec minimal early —
broad write powers expand the attack surface and muddy what the reward measures. Revisit if
a benign task genuinely requires defender writes.

---

## 4. Technology choices

| Decision | Choice | Why |
|---|---|---|
| RL algorithm | **GRPO** (group-relative) | Binary verifiable reward → drop the value network. Less VRAM, simpler than the PPO you know, field-standard for this reward shape. |
| Models | **Qwen2.5 family, asymmetric** | Defender small (0.5–3B), attacker larger (7B). Cheap, strong, supported across every RL framework. |
| Fine-tuning | **LoRA / QLoRA** | Budget-critical: train adapters, not full weights. |
| Framework (early) | **TRL `GRPOTrainer`** | Fastest to learn (Phases 0–2). Great docs, minimal ceremony. |
| Framework (later) | **verl** (or verifiers/prime-rl, RAGEN, ART) | Multi-turn concurrent self-play (Phases 4–5) outgrows TRL. Don't over-commit early. |
| Inference serving | **vLLM** | Standard for fast rollout generation while training. |

---

## 5. Phased plan

Each phase lists: **Goal · Tasks · Exit criteria · Rough compute.** Phases 0–1 need **zero RL
training** — we build the whole environment and eval instrument while ramping the LLM stack,
then hit training when ready.

### Phase 0 — De-risk the LLM-RL stack on a toy task
**Goal:** one GRPO run working end-to-end before the sandbox exists — the "hello world" that
converts classical-RL intuition into LLM-RL mechanics (token-level reward, KL-to-reference,
LoRA, vLLM rollouts).

Tasks:
- [ ] Stand up TRL + vLLM + LoRA on the available GPU; confirm a small model loads and serves.
- [ ] Pick a trivial verifiable-reward task (e.g. arithmetic/countdown): reward = exact-match.
- [ ] Run GRPO on Qwen2.5-1.5B until reward moves; log KL, reward, response length.
- [ ] Write a one-page "LLM-RL mechanics" note for ourselves (the gotchas we hit).

**Exit:** reward curve goes up on the toy task; you understand every knob in the loop.
**Compute:** minimal — a few GPU-hours; can use rented single GPU or free Colab/Kaggle.

### Phase 1 — Sandbox + benchmark (the v1 deliverable)
**Goal:** a standalone, publishable agentic-safety benchmark. Inference-only. This ships
regardless of whether RL ever converges.

Tasks:
- [ ] Sandbox container: tool set (read files, run limited commands), canary secrets, template
      reset. Harness that runs an episode loop and enforces the defender gate.
- [ ] Environment/policy separation: attacker + defender as vLLM endpoints the harness calls.
- [ ] **Attack task set** with machine-checkable success predicates (canary match). Cover the
      four categories: credential exfiltration, PII leak, system-prompt extraction, tool-
      permission escalation.
- [ ] **Benign task set** with verifiable completion targets (legitimacy credential + expected
      artifact). Include *surface-suspicious-but-justified* cases.
- [ ] Fixed *prompted* (untrained) baseline attacker + defender.
- [ ] **Frozen held-out eval harness** + metrics: attack-success-rate (ASR), benign-completion
      rate, over-refusal rate. Wire in AgentDojo / InjecAgent as external held-out eval.
- [ ] Baseline report: run several off-the-shelf models as attacker and as defender; publish
      the leaderboard.

**Exit:** we can drop any model in as attacker or defender and get ASR + benign-completion +
over-refusal numbers, including on external benchmarks.
**Compute:** inference-only. ~\$50–200 (or near-free with small local models).

### Phase 2 — Train defender vs. frozen attacker
**Goal:** first real RL training. Non-stationarity removed so we can debug sparse reward +
over-refusal penalty in isolation.

Tasks:
- [ ] Define the defender reward: `-leak + benign_completion - over_refusal` (weights tuned so
      false positives cost as much as false negatives).
- [ ] Start with a **tight turn budget** (1–2 exchanges); grow only once stable.
- [ ] GRPO + LoRA on the small defender against a fixed pool of prompted attackers.
- [ ] Track: held-out ASR ↓, benign-completion held ≥ target, over-refusal not exploding.

**Exit:** trained defender beats the prompted-baseline defender on held-out attacks *without*
tanking benign-completion.
**Compute:** first real spend — order \$100s of GPU-time.

### Phase 3 — Train attacker vs. frozen defender
**Goal:** validate the other side of the loop.

Tasks:
- [ ] Attacker reward = leak achieved (with shaping/partial credit to fight sparsity over long
      rollouts).
- [ ] GRPO + LoRA on the attacker against the best frozen Phase-2 defender.
- [ ] Offline judge labels technique per episode → first diversity graph.

**Exit:** trained attacker raises ASR against the frozen defender; techniques are diverse, not
collapsed to one exploit.

### Phase 4 — Alternating frozen self-play
**Goal:** de-risk instability before concurrent updates (fictitious self-play).

Tasks:
- [ ] Alternate: freeze one side, train the other to convergence, swap. Repeat.
- [ ] Add checkpoints to the opponent pool each round; sample opponents from the pool.
- [ ] Watch for cycling (regression against previously-beaten strategies).

**Exit:** several alternation rounds with monotone-ish improvement on the frozen held-out eval.

### Phase 5 — Concurrent self-play + pools + diversity
**Goal:** the full system: both sides updating, growing pools, technique diversity maintained.

Tasks:
- [ ] Concurrent updates (likely on verl by now).
- [ ] Pool management: reservoir of checkpoints across skill/technique; diversity-aware
      opponent sampling driven by the offline judge's technique labels.
- [ ] Strategy-over-time analytics (technique graph).

**Exit:** stable concurrent training; defender generalizes to held-out + external benchmarks.

### Stretch (explicitly not v1)
- **CoT-distillation SFT (deliberative alignment).** Periodically take failed episodes;
  generate a corrected defender CoT that cites the relevant constitution clause and would have
  caught the attack; **verify against ground-truth + judge before using as SFT data** (no
  unverified self-reflection). This is also where the attacker-CoT teacher signal lives.
  Constitution text is used only at data-generation time, internalized into weights — not held
  in the defender's context at deployment.
- **Self-improving constitution.** Judge proposes new/refined clauses from recurring uncovered
  failure patterns; folded into the next SFT round.
- **Open-sourcing** the benchmark and defender, framed as one defensive layer, evaluated on
  external held-out benchmarks.
- **Parked question (not pursued now):** could the same framework train a more-capable,
  less-trusted model to *earn* trust over time, rather than keeping the monitor permanently
  smaller/separate?

---

## 6. Evaluation & success metrics

Measured on the **frozen held-out eval** (frozen attackers + frozen benign set + external
benchmarks) on a fixed cadence — internal win-rates alone don't tell us if we're improving vs.
cycling.

- **Attack-success-rate (ASR):** fraction of attack tasks where the canary leaked. ↓ is better.
- **Benign-completion rate:** fraction of benign tasks where the required artifact was produced.
- **Over-refusal rate:** benign tasks blocked/failed despite valid credentials.
- **Headline success claim (falsifiable):** trained defender reduces held-out ASR by **X%**
  while holding benign-completion **≥ Y%** (set X, Y before Phase 2 runs).
- **Transfer:** performance on AgentDojo / InjecAgent it was never trained against.

---

## 7. Compute / budget tiers

Budget is uncertain (\$500–\$10k). Plan degrades gracefully:

- **~\$500:** Phase 0 + full Phase 1 benchmark (inference-only) + **one** Phase-2 defender-only
  LoRA run on a 0.5–1.5B model. Deliverable: benchmark + a first trained-defender result.
- **~\$10k:** Phases 2–4 meaningfully on ≤7B models, LoRA, short horizons. Deliverable:
  benchmark + validated two-sided self-play.
- **Benchmark (Phase 1) is inference-only and cheap — it ships at any funding level.**

---

## 8. Risks & open questions

- **Sparse + long-horizon reward.** Single-bit leak/no-leak over a multi-turn rollout is hard
  credit assignment. Mitigations: short horizons first, reward shaping / partial credit,
  curriculum. (This is the core RL difficulty; your classical-RL background is the asset here.)
- **Benign-set quality is the sleeper variable.** Whether the defender learns "allow legit /
  block malicious" vs. "refuse everything" is mostly determined by the benign set. Treat its
  construction as a first-class workstream, not an afterthought.
- **Degenerate self-play equilibria** (attacker/defender converge on a "safe word" pattern;
  attack distribution collapses). Mitigations: pools + diversity-aware sampling + the offline
  judge's technique tracking.
- **Rollout throughput** if reset isn't cheap enough → revisit env-state reset strategy.
- **Open:** exact reward weights (leak vs. over-refusal); turn-budget schedule; when to migrate
  TRL → verl; how large a benign set is "enough" (and generate-on-the-fly vs. frozen).

---

## 9. Working style

- We build **collaboratively, piece by piece.** Each phase gets its own detailed
  implementation plan (via the writing-plans workflow) *when we start it* — this document is
  the map, not the turn-by-turn.
- Phase gates are real: we don't start Phase N+1 until Phase N's exit criteria are met.
- Every task/dataset item ships with a machine-checkable success predicate — that's the spec.
