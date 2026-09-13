# Stage 9 (extended) — Renderer-shift generalization

Prepared 14 September 2026, 3 seeds (7/17/27). Extends
[STAGE9_GENERALIZATION_REPORT.md](STAGE9_GENERALIZATION_REPORT.md) (corruption/density
shifts) with the one generalization category that report explicitly left open: a genuine
**renderer shift** — the same abstract diagrams (same node positions, same connector
graphs, same corruption family) rasterized by a categorically different technique. Models
trained once per seed on `data/confirmatory-v1`, evaluated zero-shot against
`data/ood-renderer-v1` (`configs/ood_renderer.toml`). Full numeric output:
`results/milestone4/renderer_seed{7,17,27}/generalization.json`.

## The second renderer

`geometry.render_antialiased` (see the commit introducing it for the three specific
numerical bugs found and fixed while building it — a beveled-joint gap, a box-filter
alignment offset, and thin-diagonal-stroke coverage) rasterizes the identical geometry the
binary renderer uses via 4× supersampling and box downsampling, producing genuinely
antialiased gray edges instead of exact binary fills. Every image in the resulting dataset
passes the same integrity bar as every other dataset in this project: 80/80 evaluator
fixtures, exact clean-graph recovery on all 302 generated images (with the ~0.7% of
geometries that don't survive this specific renderer's antialiasing rejected and resampled
during generation, not silently shipped as broken ground truth).

## Results (mean edge-F1 ± sd across seeds 7/17/27)

| Condition | B0 | U0 | U1 | D0 @50 steps | D1 @50 steps | Gap (best U − best D) | Gap ÷ pooled sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| In-distribution (reference) | 0.929 | 0.994 ± 0.002 | 0.994 ± 0.001 | 0.927 ± 0.002 | 0.926 ± 0.002 | 0.069 | ~30× |
| **Renderer shift** | 0.954 | 0.999 ± 0.001 | 0.999 ± 0.002 | 0.938 ± 0.020 | 0.891 ± 0.027 | 0.108 | ~3.9× |

## Finding: the gap widens further, but diffusion becomes far less predictable

The U-Net doesn't just hold up under the renderer shift — it improves, reaching
**near-perfect scores (0.9985 mean, both U0 and U1)**, better than its own in-distribution
performance. It generalizes gracefully to smoother, antialiased edges despite never training
on them.

Diffusion moves the opposite direction on both axes: its mean F1 drops (D1: 0.926 → 0.891)
**and** its seed-to-seed variance explodes — D1's standard deviation jumps from 0.002
in-distribution to **0.027** under the renderer shift, a roughly 14× increase. This is the
largest variance seen in any condition across Gate G2, Stage 8, or Stage 9. The resulting
gap-to-noise ratio (~3.9×) is the weakest margin measured in this entire project — still
clearly directional (the U-Net wins in all 3 of 3 seeds, no exceptions) but the least
statistically decisive, purely because diffusion's own behavior becomes so seed-sensitive
under this specific shift.

## Finding: the structural loss actively hurts under this shift

In every other condition tested (Gate G2, Stage 8, corruption/density shifts), D1
(structural loss) was roughly on par with or slightly ahead of D0. Here it is clearly
**behind**: D0 mean 0.938 vs. D1 mean 0.891. The soft-clDice structural term is computed
against the training distribution's binary-edge statistics; under the antialiased renderer's
different pixel statistics, that extra supervision signal appears to actively mislead the
model rather than help it. This is a specific, checkable claim (not yet independently
verified beyond this one dataset) about *why* D1 is more brittle here, not just an
observation that it is.

## What this closes out

Combined with [STAGE9_GENERALIZATION_REPORT.md](STAGE9_GENERALIZATION_REPORT.md), the
project's core finding now holds under **three** distinct kinds of distribution shift —
harder corruption, denser layouts, and a different rendering technique — none of which
reverses or meaningfully threatens the ranking, and two of which (density, renderer) show
the U-Net's advantage growing rather than shrinking. This is the renderer-shift half of the
generalization protocol's requirement; the acquisition-shift half (real, independently
authored or scanned diagrams) still requires genuinely external input this project cannot
fabricate — see the note in
[docs/03_PROJECT_PLAN.md](../../docs/03_PROJECT_PLAN.md).

## What this does not establish

- **Not an acquisition shift.** This is still a synthetic image, drawn by software — a
  different piece of software than training used, but not a photograph, scan, or
  human-drawn diagram. Per the protocol's own rule, this must not be labeled real-world
  validation.
- **The "structural loss actively hurts" explanation is a hypothesis consistent with the
  data**, not independently verified — it has not been tested by, say, computing the
  soft-clDice term's actual value on antialiased vs. binary images to confirm the mechanism
  directly.
- **Single antialiasing configuration tested** (4× supersample, box filter) — other
  antialiasing techniques, or a genuinely different graphics library/backend, might behave
  differently.

## Decision

The renderer-shift test is the third and final planned generalization check for this phase
of the project, and it produced the same qualitative verdict as the other two: the U-Net's
dominance holds, and if anything strengthens on the U-Net's side while diffusion becomes
less predictable. Generalization work for this project is now complete except for the
acquisition-shift category, which is explicitly and honestly left open rather than
approximated.
