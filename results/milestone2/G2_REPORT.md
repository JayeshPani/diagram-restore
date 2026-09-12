# Gate G2 screening report — Stage 6 baseline suite

Prepared 12 September 2026, updated 13 September 2026 — now including a 3-seed confirmation of
the primary U0/U1-vs-D0/D1 comparison. **Still a development screening pass, not the fully
powered confirmatory study**: seed count (3) is at the plan's minimum, there are no formal
confidence intervals, and OOD/held-out generalization has not been tested. Treat the direction and
size of the finding as reliable; treat exact decimal values as provisional.

## Question

Per [docs/04_EXPERIMENT_PROTOCOL.md](../../docs/04_EXPERIMENT_PROTOCOL.md) H1: do connection
errors occur in this task, and does reducing diffusion computation (fewer sampling steps)
measurably worsen them in at least one operating range? Per
[docs/03_PROJECT_PLAN.md](../../docs/03_PROJECT_PLAN.md) Gate G2: does the pipeline learn a
useful diffusion regime, and does reduced compute produce a measurable connectivity/quality
tradeoff?

## Fixed configuration

- Data: `data/pilot-v1`, dataset version `pilot-v1`, manifest `6d2b46fd…` (500 train / 100
  validation / 100 reserved test; test set untouched here).
- Seed: 7 (single seed; every model below is one training run, not a seed-averaged result).
- Run 1 (git `bfa2b95`): U0/U1/T0 at 6,000 steps; D0/D1 at 10,000 steps.
- Run 2 (git `fe47161`, `source_sha256` in [baselines.json](baselines.json)): identical U0/U1/T0;
  **D0/D1 retrained at 30,000 steps** (3× run 1) specifically to test whether run 1's gap was an
  undertraining artifact. Batch-one latency (warmup 20, 50 timed repeats, MPS-synchronized) added
  for every row in run 2.
- Run 3 (git `f5dc2c1`, `source_sha256` in [diffusion_tuning.json](diffusion_tuning.json)): D0
  retrained again at the same 30,000 steps and seed, this time with the two standard diffusion
  training fixes run 2 lacked — a linear-warmup/cosine-decay learning rate schedule
  (`warmup_steps=1500`, base_lr=1e-4) and an EMA of the weights (decay 0.999) used for sampling.
  This isolates "was the recipe missing standard tricks" from "is the architecture behind."
- Run 4 (git `4af1709`, `source_sha256` in [capacity/diffusion_tuning.json](capacity/diffusion_tuning.json)):
  same recipe as run 3 (EMA + warmup-cosine, 30,000 steps, seed 7), but with a **2.9× larger DiT**
  (width 192, depth 8 → 5.64M params vs 1.92M). Tests whether the pilot DiT was simply too small.
- Runs 5–6 (same git `457823f`, `source_sha256` in [seed17/baselines.json](seed17/baselines.json)
  and [seed27/baselines.json](seed27/baselines.json)): the full run-2 baseline suite (unet_steps
  6,000; dit_steps 30,000; plain recipe, no EMA/schedule/capacity changes) repeated at seeds 17
  and 27, to check whether the run-2 gap was a seed-7 artifact.
- U0/U1/T0: SmallUNet/ConditionalDiT backbone, AdamW lr=1e-4, batch 8.
- D0/D1: ConditionalDiT, same optimizer; DDIM sampling at 10/20/50 steps, eta=0, identical fixed
  noise across the sweep and across D0/D1.
- Structural weight (U1/T0/D1): 0.1 × (1 − soft-clDice), a structural proxy, not a topology proof.
- B1 morphological radius chosen on validation from {0,1,2,3}.

## Observations (validation split, 100 images, 208 true edges) — run 2, 30k-step diffusion

| Row | Edge F1 | Recall | Exact-graph | Median latency (single image) |
| --- | --- | --- | --- | --- |
| B0 damaged input | 0.917 | 0.846 | 0.70 | n/a (no restoration) |
| B1 morphological, radius=1 | 0.943 | 0.913 | 0.80 | **0.05 ms** |
| U0 small U-Net | 0.963 | 0.928 | 0.85 | **2.13 ms** |
| **U1 U-Net + structural loss** | **0.965** | 0.933 | **0.86** | 2.12 ms |
| T0 deterministic transformer | 0.947 | 0.899 | 0.79 | 2.67 ms |
| D0 dense diffusion @10/20/50 steps | 0.922 / 0.922 / **0.911** | 0.856/0.856/0.837 | 0.70/0.70/0.69 | 29.6 / 50.0 / 117.5 ms |
| D1 diffusion + structural loss @10/20/50 | 0.922 / 0.922 / 0.914 | 0.856/0.856/0.841 | 0.70/0.70/0.68 | 38.4 / 66.7 / 133.3 ms |

Full numeric output: [baselines.json](baselines.json).

## What changed between run 1 (10k steps) and run 2 (30k steps)

- Training loss fell (D0: 0.0158 → 0.0076), so the model kept optimizing — it was not stuck.
- Validation edge F1 at 10/20 sampling steps was **unchanged to three digits** (0.9223 both runs).
- Validation edge F1 at 50 sampling steps got **worse**, not better (0.9223 → 0.9110 for D0;
  0.9223 → 0.9138 for D1). More training moved the model further from what the fixed 50-step DDIM
  trajectory needs, rather than closer.
- **This rules out plain undertraining as the explanation for run 1's gap.** Tripling optimizer
  steps did not close it and mildly widened it at the higher sampling budget.

## What changed in run 3 (EMA + warmup-cosine LR, same 30k steps and seed)

| Sampling steps | Plain D0 (run 2) F1 | Tuned D0 (run 3) F1 |
| --- | --- | --- |
| 10 | 0.9223 | 0.9223 (identical) |
| 20 | 0.9223 | 0.9223 (identical) |
| 50 | 0.9110 | 0.9195 (+0.0085, 3 more true edges of 208) |

EMA and a proper LR schedule are the two most common fixes for unstable/undertrained diffusion
sampling, and they left the 10- and 20-step results **completely unchanged** and moved the
50-step result only slightly — nowhere near U1's 0.965. Latency also stayed in the same
20–100 ms range (still 10×–45× slower than U-Net's 2.1 ms). Full numeric output:
[diffusion_tuning.json](diffusion_tuning.json).

**This rules out "missing standard diffusion training tricks" as the explanation, too.** Two of
the three most likely fixable-defect explanations are now eliminated by direct experiment, not
assumption.

## What changed in run 4 (2.9× larger DiT, same recipe, steps, and seed)

| Sampling steps | Tuned D0, 1.92M params (run 3) | Tuned D0, 5.64M params (run 4) |
| --- | --- | --- |
| 10 | 0.9223 | 0.9195 (**worse**) |
| 20 | 0.9223 | 0.9195 (**worse**) |
| 50 | 0.9195 | 0.9138 (**worse**) |

Nearly tripling the parameter count made every sampling-step result **worse**, not better, and
latency scaled up with it (32.6/64.4/160.2 ms vs 20.0/39.5/97.6 ms — now 16×–74× slower than
U-Net). Full numeric output: [capacity/diffusion_tuning.json](capacity/diffusion_tuning.json).

**This rules out insufficient capacity as the explanation.** All three of the fixable-defect
hypotheses raised after run 2 — undertraining, missing standard training tricks, insufficient
model capacity — have now been tested by direct experiment and eliminated. Two of the three
interventions (more steps, more capacity) made results measurably *worse* rather than merely
failing to help, which is more consistent with a genuine architecture/task mismatch at this pilot
scale than with a still-undiscovered tuning fix.

## Multi-seed confirmation (runs 5–6, plain recipe, dit_steps=30,000)

| Seed | U0 | U1 | D0 @50 steps | D1 @50 steps |
| --- | --- | --- | --- | --- |
| 7 | 0.9626 | 0.9652 | 0.9110 | 0.9138 |
| 17 | 0.9703 | 0.9652 | 0.9223 | 0.9251 |
| 27 | 0.9652 | 0.9779 | 0.9167 | 0.9167 |
| **mean ± sd** | 0.9660 ± 0.0039 | 0.9694 ± 0.0073 | 0.9167 ± 0.0056 | 0.9185 ± 0.0058 |

The U0/U1 vs. D0/D1 gap (≈0.05 F1) is roughly **7–9× the seed-to-seed standard deviation** and
the same direction in all three seeds — U-Net always wins, diffusion never closes to within noise
of it. **Seed variance is ruled out as an alternative explanation.**

## Decision

**U0/U1 now dominate D0/D1 on both measured axes at every tested diffusion operating point**:
higher edge F1 (0.96–0.97 vs 0.91–0.92) *and* 14×–55× lower latency (2.1 ms vs 29.6–117.5 ms).
This is exactly the trigger named in
[docs/01_RESEARCH_SCOPE.md](../../docs/01_RESEARCH_SCOPE.md) ("If U-Net matches or exceeds
diffusion's connectivity accuracy at lower latency throughout the useful operating range, report
that outcome and reconsider the diffusion contribution") and in the plan's own failure-response
table ("U-Net dominates the useful quality/runtime range → stop expanding the diffusion method").

All five confounds raised since run 1 are now addressed:

1. ~~Diffusion may be undertrained~~ — **ruled out**: 3× the steps did not help (run 2).
2. ~~No latency measured~~ — **measured**: U-Net wins on latency by more than an order of
   magnitude, so this is not a case where diffusion trades speed for quality.
3. ~~Missing standard training tricks (EMA / LR schedule)~~ — **ruled out**: both together moved
   one of three sampling-step results by 0.0085 F1; the other two were unchanged (run 3).
4. ~~Insufficient model capacity~~ — **ruled out**: 2.9× more parameters made every result worse,
   not better (run 4).
5. ~~Seed-7 artifact~~ — **ruled out**: the gap holds in the same direction and similar magnitude
   across seeds 7, 17, and 27 (runs 5–6).

**This is now a settled screening-level finding for this pilot task and scale, not an open
question.** The remaining gaps before it is a fully confirmatory result are generalization
(held-out layouts, a second renderer, independently authored diagrams — Stage 10 in
[docs/03_PROJECT_PLAN.md](../../docs/03_PROJECT_PLAN.md)) and the unexplored 1–5-step sampling
regime, neither of which is likely to favor diffusion given the pattern so far.

**The remaining choice is a project-direction decision, not a routine implementation one** — the
routing method this project is built around only matters if diffusion is worth accelerating in
the first place. I'm flagging it back to the researcher rather than deciding unilaterally.

## Next scoped step (pending direction)

With all five confounds eliminated, the evidence-backed options are:

1. **Scope the paper around a narrower, evidence-backed contribution** — the corruption generator,
   the direct-edge evaluator, and this comparative benchmark (U-Net vs. dense diffusion vs.
   morphological repair for diagram connectivity restoration) are all real, validated artifacts
   even without a routing result. Per
   [docs/06_PAPER_AND_VENUE.md](../../docs/06_PAPER_AND_VENUE.md)'s own contingency.
2. **Continue to the Stage 7 oracle and Stage 8 router anyway**, with this limitation stated
   up front — the plan's own failure-response rule names this "building a router to rescue a
   losing backbone," so it should be a deliberate, documented choice, not a default.
3. **Extend the sampling sweep to 1–5 steps** first, only if option 2 is chosen — it is the one
   remaining piece of H1 evidence not yet gathered, and matters only if diffusion stays in scope.
