# Stage 8 — Confirmatory-scale benchmark (5,000/500/1,000 diagrams, 3 seeds)

Prepared 13 September 2026. Confirms Gate G2 and the Stage 7 appearance-divergence finding
at a 9.3× larger, independently generated dataset. Not the pilot split reused at a larger
size — `data/confirmatory-v1` is a fresh generation (seed 20260913, vs. pilot-v1's
20260912; see [../../configs/confirmatory.toml](../../configs/confirmatory.toml)), audited
separately (`results/milestone3/integrity.json`: 80/80 fixtures, exact clean-graph recovery
on all 6,500 images).

Same hyperparameters as the pilot-scale runs throughout: unet_steps=6000, dit_steps=30000,
sampling_steps={10,20,50}, structural_weight=0.1, latency 20-warmup/50-repeat. Full numeric
output: `results/milestone3/seed{7,17,27}/baselines.json`.

## Headline: the gap didn't just replicate — it widened

| Row | Pilot scale (700 diagrams) mean F1 | Confirmatory scale (6,500 diagrams) mean F1 |
| --- | --- | --- |
| U0 | 0.966 | **0.9944** |
| U1 | 0.969 | **0.9944** |
| D0 @50 steps | 0.917 | 0.9269 |
| D1 @50 steps | 0.919 | 0.9257 |

More training diversity (10× the unique training images, same 6,000/30,000 optimizer
steps) let the U-Net converge to near-ceiling performance (0.994 mean F1, sd 0.001–0.002),
while dense diffusion barely moved (0.926–0.927, sd ~0.002). **The accuracy gap widened
from ~0.05 F1 at pilot scale to ~0.069 F1 at confirmatory scale**, and — because both
sides got *more* consistent, not less — the gap-to-noise ratio went from ~7–9× the
seed-to-seed standard deviation at pilot scale to **roughly 29×** here.

This says something specific: within a *fixed optimizer-step budget*, the small U-Net's
simpler function class extracts far more value from additional training diversity than the
diffusion model's harder optimization problem (denoising across 1,000 timesteps) does. This
is not "diffusion never improves with more data" — it is "at this step budget, on this
task, it didn't."

## Latency confirms too

U0/U1 median latency: 2.10 ms (sd 0.03 ms across seeds) — essentially unchanged from pilot
scale, as expected (latency depends on model size and sampling procedure, not training set
size). D0 @50 steps: 96.2 / 97.3 / 160.1 ms across seeds 7/17/27 — the seed-27 measurement
is a clear outlier (likely system load/thermal variation during that specific run, not a
model difference, since D0@50's edge-F1 and Dice were unremarkable for that seed) and
should be re-measured before being treated as a real latency increase. Even taking it at
face value, U-Net remains **46×–76× faster**.

## Appearance-vs-edge divergence also strengthens

| Seed | Dice range | Edge-F1 range | Ratio | B0 beats every diffusion config on Dice? |
| --- | --- | --- | --- | --- |
| 7 | 0.0096 | 0.0698 | 7.3× | Yes (6/6) |
| 17 | 0.0095 | 0.0712 | 7.5× | Yes (6/6) |
| 27 | 0.0119 | 0.0749 | 6.3× | Yes (6/6) |

Compression ratio is now ~6–7.5× (vs. ~3.8–4.9× at pilot scale) — the compression got
*worse*, not better, at the larger scale, because the U-Net's near-ceiling Dice (0.9996–
0.9998) leaves almost no room for Dice to separate it from anything, while its edge-F1
advantage kept growing. The untouched damaged input again beats every diffusion
configuration on Dice, now 18/18 confirmatory-scale configs on top of the 18/18 from the
pilot scale — **36 out of 36 checked configurations, across two independently generated
datasets at two different scales.**

## What this settles

- **Criterion C** (docs/04_EXPERIMENT_PROTOCOL.md §10): met, more strongly than at pilot
  scale. The U0/U1-vs-D0/D1 ranking is not just seed-stable but scale-stable, and the
  effect size grows with more data rather than shrinking — the opposite of what "the pilot
  result was noise/overfit to a small dataset" would predict.
- **RQ3** (docs/04_EXPERIMENT_PROTOCOL.md §1): "does the RQ1 finding replicate at
  confirmatory scale" — **yes**, on a freshly generated, non-overlapping dataset.
- The Stage 7 evaluator-divergence finding (Finding 04) is now confirmed on **two
  independent datasets at two scales**, not just three seeds of one dataset.

## What remains open

- **RQ4 / Stage 9 (generalization):** held-out layout families, a second renderer, or
  independently authored diagrams are still untested — this confirmatory split is a larger
  sample from the *same* generator distribution, not a distribution shift.
- The seed-27 D0@50 latency outlier should be re-measured with more repeats or on an idle
  machine before being reported as a real number in the paper.
- Whether giving diffusion a larger optimizer-step budget (rather than more data) at this
  scale would close the gap was not tested here — Gate G2's step-budget check (10k vs. 30k)
  was run at pilot scale only.

## Decision

The confirmatory scale did not just fail to overturn the pilot finding — it produced a
**larger, more statistically decisive version of the same result**, on independently
generated data. Combined with the four fixable-defect checks from Gate G2 and the two-scale
confirmation of the evaluator-divergence finding, this is now strong, multi-angle evidence
for both of the project's core claims. The next open item is generalization (Stage 9), not
further confirmation at this scale.
