# Stage 9 — Generalization to held-out distribution shifts

Prepared 13 September 2026, 3 seeds (7/17/27). Models trained once per seed on
`data/confirmatory-v1`'s training split (5,000 diagrams), then evaluated **zero-shot, with
no retraining or re-tuning**, against three conditions: the in-distribution validation
split (reference), and two held-out test sets generated with shifted parameters the models
never saw during training. Full numeric output:
`results/milestone4/seed{7,17,27}/generalization.json`.

## The two distribution shifts tested

Per `docs/04_EXPERIMENT_PROTOCOL.md` §9's list ("unseen layout families... stronger yet
interpretable corruptions, and crowded near-misses"), scoped to shifts that fit the
existing, already-validated generator and evaluator without any code changes to either:

- **`ood_corruption`** (`configs/ood_corruption.toml`, fresh seed 20260914, 300 test images):
  2× the noise and blur severity used in training (noise_sigma_max 0.03→0.06, blur_sigma_max
  0.6→1.2), longer line gaps (max_gap_length 3→5), and corruption weights shifted to
  emphasize the hardest combined case (`gap`, which stacks noise+blur+the gap itself) at the
  expense of the `clean` stratum.
- **`ood_density`** (`configs/ood_density.toml`, fresh seed 20260915, 300 test images):
  always exactly 5 nodes (the training distribution's maximum) instead of the trained 2–5
  range — the densest, most crowded layouts the domain supports, stress-testing near-misses
  between nearby unrelated shapes.

Both datasets pass the same integrity checks as every other dataset in this project (80/80
evaluator fixtures, exact clean-graph recovery on every clean-rendered image) before being
used for evaluation.

## Results (mean edge-F1 ± sd across seeds 7/17/27)

| Condition | B0 (no restoration) | U0 | U1 | D0 @50 steps | D1 @50 steps | Gap (U1 − D1) | Gap ÷ pooled sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| In-distribution | 0.929 | 0.994 ± 0.002 | 0.994 ± 0.001 | 0.927 ± 0.002 | 0.926 ± 0.002 | 0.069 | ~30× |
| `ood_corruption` | 0.815 | 0.928 ± 0.003 | 0.929 ± 0.012 | 0.832 ± 0.006 | 0.836 ± 0.007 | 0.092 | ~6.6× |
| `ood_density` | 0.953 | 0.994 ± 0.001 | 0.996 ± 0.002 | 0.951 ± 0.000 | 0.950 ± 0.001 | 0.046 | ~22.5× |

(B0's sd is exactly 0 in every condition — it depends only on the fixed damaged images, not
on model training seed, as expected.)

## Finding: the U-Net's dominance survives both shifts, with no exceptions

In every one of the three conditions — including the two the models never trained on — U0
and U1 clearly beat D0 and D1, by a wide margin relative to seed-to-seed noise. The margin
narrows under the harder corruption shift (gap÷sd drops from ~30× to ~6.6×, since diffusion
in particular becomes more seed-variable under harsher degradation — U1's sd rises to 0.012,
4–10× its sd elsewhere), but it never comes close to closing. **No condition in this test
flips or even meaningfully narrows the ranking established at pilot and confirmatory
scale.**

## What each shift specifically shows

- **`ood_corruption`** is the more informative stress test: absolute performance drops
  substantially for every method (B0 falls to 0.815, a 12-point drop from in-distribution),
  and the gap between U-Net and diffusion actually **widens in absolute terms** (0.069 →
  0.092 F1) even as it narrows relative to the now-larger noise. This says the U-Net's
  advantage is not a fragile artifact of the exact training corruption severity — it holds
  under corruption twice as harsh as anything trained on.
- **`ood_density`** is a milder shift by construction (training already included up to 5
  nodes; this just makes 5 nodes universal instead of one of five possibilities), and the
  results reflect that — absolute performance for every method stays close to
  in-distribution levels. The gap still holds clearly (~22.5× pooled sd), showing the
  advantage isn't specific to less-crowded layouts either.

## What this does and doesn't establish

**Established:** the core finding (small U-Net beats dense diffusion on this restoration
task) is not an artifact of the exact training distribution — it survives a 2× harder
corruption regime and a maximally dense layout regime, both evaluated zero-shot on models
that never saw either during training.

**Not established:**
- **Not a renderer or acquisition shift.** Both OOD sets use the identical rendering
  pipeline, node shapes, and connector style as training — only corruption parameters or
  node count changed. A second synthetic renderer or real/independently authored diagrams
  (also named in the protocol) were not tested; that remains the one distribution-shift
  category this project has not touched.
- **Not a junction/topology-convention shift.** Free-standing branch points and higher-degree
  graph conventions were explicitly excluded, per the protocol's own requirement for an
  extended, separately-tested evaluator before attempting them.
- **Single training run per seed** — these are the same trained models from a single
  training pass per seed, evaluated against multiple test sets, not independently retrained
  per condition. This is intentional (the whole point is testing one fixed model's
  transfer), not a shortcut.

## Decision

Both distribution shifts tested support the same conclusion as the in-distribution and
confirmatory-scale results: for diagram connectivity restoration on this domain, a small
U-Net dominates dense diffusion, and that dominance is not narrowly overfit to the exact
training distribution. Combined with Gate G2's four confound checks, the confirmatory-scale
replication (Stage 8), and the two-scale appearance-metric divergence (Stage 7), this
project's two core empirical claims now rest on evidence from two independently generated
in-distribution datasets, two scales, and two out-of-distribution shifts — a substantially
more thorough evidence base than the original single-pilot-run premise this project started
from.

**What would still strengthen this further** (not attempted here, flagged for future work
per the protocol's own list): a second synthetic renderer, or real/independently authored
diagrams, to test renderer and acquisition shift rather than only parametric shifts within
the same generator.
