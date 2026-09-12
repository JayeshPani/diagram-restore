# Gate G2 screening report — Stage 6 baseline suite

Prepared 12 September 2026. **Single-seed development screening, not a confirmatory result.**
No confidence intervals, no repeated seeds, no measured latency yet. Do not cite these numbers
as final paper evidence; see "Next scoped step" for what turns this into one.

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
- Code: git revision `64bcf0c6` on branch `codex/research-plan`; `source_sha256` recorded inside
  `baselines.json`.
- Seed: 7 (single seed; every model below is one training run, not a seed-averaged result).
- U0/U1/T0: SmallUNet/ConditionalDiT backbone, AdamW lr=1e-4, batch 8, 6,000 steps.
- D0/D1: ConditionalDiT, same optimizer, 10,000 steps; DDIM sampling at 10/20/50 steps, eta=0,
  identical fixed noise across the sweep and across D0/D1.
- Structural weight (U1/T0/D1): 0.1 × (1 − soft-clDice), a structural proxy, not a topology proof.
- B1 morphological radius chosen on validation from {0,1,2,3}.
- Full numeric output: [baselines.json](baselines.json).

## Observations (validation split, 100 images, 208 true edges)

| Row | Description | Edge F1 | Precision | Recall | Exact-graph | Invented/img |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | Damaged input, untouched | 0.917 | 1.000 | 0.846 | 0.70 | 0.00 |
| B1 | Morphological closing, radius=1 | 0.943 | 0.974 | 0.913 | 0.80 | 0.05 |
| U0 | Small U-Net | 0.963 | 1.000 | 0.928 | 0.85 | 0.00 |
| U1 | U-Net + structural loss | **0.965** | 1.000 | 0.933 | **0.86** | 0.00 |
| T0 | Deterministic transformer + structural loss | 0.947 | 1.000 | 0.899 | 0.79 | 0.00 |
| D0 @10/20/50 steps | Dense diffusion | 0.919 / 0.922 / 0.922 | 1.000 | 0.851 / 0.856 / 0.856 | 0.69 / 0.70 / 0.70 | 0.00 |
| D1 @10/20/50 steps | Dense diffusion + structural loss | 0.922 / 0.922 / 0.922 | 1.000 | 0.856 / 0.856 / 0.856 | 0.70 / 0.70 / 0.70 | 0.00 |

## Representative successes / failures

- Every trained model has **zero invented edges** on validation; B1 is the only row that invents
  edges (over-closing merges nearby parallel lines — the false-positive mechanism the protocol
  anticipated for morphological repair).
- U0/U1 clearly separate from B0/B1 and from D0/D1: the small U-Net recovers 193–194 of 208 true
  edges; dense diffusion recovers only 177–178 at any tested step count.
- T0 (same transformer backbone as D0/D1, one deterministic pass, structural loss) reaches 187/208
  — better than dense diffusion, worse than the U-Net. This is evidence *against* "transformer
  capacity alone explains a U-Net gap" (there is no U-Net gap to explain yet at this budget), and
  weak evidence that the convolutional U-Net's inductive bias, not model class, is doing the work
  here.
- D0 and D1 are statistically indistinguishable from each other (single seed) and are flat across
  10 → 50 sampling steps: recall does not move outside ±1 edge. **H1's "fewer steps worsens
  connectivity" is not supported in this 10–50 step range at this training budget.**

## Uncertainty

Single seed (7), single training run per row — no variance estimate exists yet. "Loss" values are
optimizer training loss, not held-out likelihood. Latency has not been measured for any row; all
comparisons above are quality-only. The flat D0/D1 step-count curve is consistent with either
(a) a genuine plateau in this operating range, or (b) both models being undertrained relative to
U0/U1 (10,000 diffusion steps ≈ 62 epochs over 500 images with one random timestep supervised per
step, versus U-Net's full-image supervision every step) — the data here cannot distinguish these.

## Compute consumed

~15 minutes wall-clock (measured, single run, M4 Pro, MPS) for the entire suite: five model
trainings, the morphological radius search, and the D0/D1 sampling sweep over 100 validation
images. Well inside the Stage 6 development ceiling in
[docs/05_ENVIRONMENT_AND_BUDGET.md](../../docs/05_ENVIRONMENT_AND_BUDGET.md).

## Decision

**Not yet a Gate G2 pass or fail — this is a single-seed screening run that surfaces a real
open question rather than settling it.** Per the plan's failure-response table, "U-Net dominates
the useful quality/runtime range" is the trigger to stop expanding the diffusion method — and on
raw numbers alone, U0/U1 already beat D0/D1 here. But two confounds must be ruled out first,
per the plan's instruction not to silently favor either outcome:

1. **Diffusion may be undertrained relative to the U-Net**, not architecturally worse. D0/D1 used
   10,000 steps against the protocol's own 20,000-step illustrative budget; U0/U1 needed only
   6,000 to plateau (loss ~0.001) — that asymmetry alone could explain the entire gap.
2. **No latency has been measured.** Gate G2 and Criterion A/B both require a runtime axis; a
   U-Net win on accuracy alone does not resolve the practical-relevance question in
   [docs/01_RESEARCH_SCOPE.md](../../docs/01_RESEARCH_SCOPE.md).
3. **The 10–50 step range may already be saturated** for this dataset; H1's premise ("fewer
   steps worsens connectivity") is untested below 10 steps.

## Next scoped step

1. Re-run D0/D1 with substantially more optimizer steps (e.g. 30,000–50,000) and confirm training
   loss has plateaued before re-comparing to U0/U1 — this directly tests confound (1).
2. Extend the sampling-step sweep down to 1–5 steps to properly probe H1's premise before
   concluding the operating range is flat.
3. Measure batch-one end-to-end latency for every row per
   [docs/04_EXPERIMENT_PROTOCOL.md §8](../../docs/04_EXPERIMENT_PROTOCOL.md) before any
   accuracy/latency tradeoff claim.
4. Only after (1)–(3): repeat the retained comparisons across seeds {7, 17, 27} per the plan's
   staged comparison budget, and make the Gate G2 call in a revised version of this report.
