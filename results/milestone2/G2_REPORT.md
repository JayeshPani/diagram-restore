# Gate G2 screening report — Stage 6 baseline suite

Prepared 12 September 2026, updated after a second run the same day. **Single-seed development
screening across two independent training budgets, not a multi-seed confirmatory result.** No
confidence intervals yet. Do not cite these numbers as final paper evidence; see "Next scoped
step" for what turns this into one.

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

## Decision

**U0/U1 now dominate D0/D1 on both measured axes at every tested diffusion operating point**:
higher edge F1 (0.96–0.97 vs 0.91–0.92) *and* 14×–55× lower latency (2.1 ms vs 29.6–117.5 ms).
This is exactly the trigger named in
[docs/01_RESEARCH_SCOPE.md](../../docs/01_RESEARCH_SCOPE.md) ("If U-Net matches or exceeds
diffusion's connectivity accuracy at lower latency throughout the useful operating range, report
that outcome and reconsider the diffusion contribution") and in the plan's own failure-response
table ("U-Net dominates the useful quality/runtime range → stop expanding the diffusion method").

Both of the confounds raised after run 1 are now addressed:

1. ~~Diffusion may be undertrained~~ — **ruled out**: 3× the steps did not help.
2. ~~No latency measured~~ — **measured**: U-Net wins on latency by more than an order of
   magnitude, so this is not a case where diffusion trades speed for quality.

The one confound not yet addressed is single-seed variance (seed 7 only) and an unexplored
1–5-step sampling regime; see below. Given how large and directionally consistent the gap is
across two independent training budgets, seed variance alone is very unlikely to reverse this
conclusion, but it is not yet formally ruled out.

**This is a project-direction decision, not a routine implementation one** — the routing method
this project is built around only matters if diffusion is worth accelerating in the first place.
I'm flagging it back to the researcher rather than deciding unilaterally which way to take it.

## Next scoped step (pending direction)

1. Confirm with seeds {17, 27} that the U-Net-dominates finding is not a seed artifact (cheap:
   ~5 more minutes per seed at this pilot scale).
2. Extend the D0/D1 sampling sweep down to 1–5 steps to fully characterize H1's premise.
3. Rule out a fixable diffusion-training defect before treating the gap as architectural: try a
   cosine LR schedule / warmup, EMA weights, and a wider/deeper DiT at matched compute, since the
   current recipe is a minimal pilot default, not a tuned diffusion baseline.
4. Depending on (1)–(3): either scope the paper around a narrower, evidence-backed restoration/
   evaluation contribution (per
   [docs/06_PAPER_AND_VENUE.md](../../docs/06_PAPER_AND_VENUE.md)'s own contingency), or identify
   a diffusion configuration that is competitive before continuing to the Stage 7 oracle and
   Stage 8 router — building a router to rescue a losing backbone is explicitly against the plan's
   own failure-response rule.
