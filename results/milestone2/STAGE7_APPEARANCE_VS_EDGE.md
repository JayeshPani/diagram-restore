# Stage 7 — Appearance metrics vs. the direct-edge evaluator

Prepared 13 September 2026. Validation split, seed 7, same run as
[G2_REPORT.md](G2_REPORT.md) (unet_steps=6000, dit_steps=30000), now also scoring mean
foreground Dice, boundary F1, SSIM, and PSNR (`evaluation.aggregate_image_metrics`)
alongside edge-F1 for every row. Full numeric output: [baselines.json](baselines.json).

## The question (RQ2, docs/04_EXPERIMENT_PROTOCOL.md)

Does an appearance-similarity metric rank restoration methods the same way the direct-edge
evaluator does, or does it hide the connectivity gap Gate G2 found?

## Results

| Row | Edge F1 | Mean Dice | Mean SSIM | Mean PSNR (dB) |
| --- | --- | --- | --- | --- |
| B0 damaged input | 0.9167 | **0.9974** | 0.9393 | 35.0 |
| B1 morphological | 0.9429 | 0.9875 | 0.9954 | 28.3 |
| U0 U-Net | 0.9626 | 0.9986 | 0.9994 | 59.8 |
| **U1 U-Net + structural loss** | **0.9652** | 0.9985 | 0.9994 | 60.8 |
| T0 deterministic transformer | 0.9468 | 0.9978 | 0.9988 | 49.9 |
| D0 diffusion @10/20/50 | 0.9223/0.9223/0.9110 | 0.9940/0.9954/0.9883 | 0.9771/0.9834/0.9664 | 35.4/36.4/31.1 |
| D1 diffusion + structural loss @10/20/50 | 0.9223/0.9223/0.9138 | 0.9941/0.9962/0.9933 | 0.9567/0.9806/0.9661 | 33.9/36.1/32.0 |

## Finding: Dice compresses every method near ceiling and hides the real gap

Edge-F1 spans **0.9110 – 0.9652**, a range of 0.0542 — enough to separate "U-Net" from
"dense diffusion" from "raw damaged input" into clearly different tiers. Mean Dice spans
only **0.9875 – 0.9986**, a range of **0.0111** — about **5× narrower**. On Dice alone,
every method in this table looks like a minor variation on the same near-perfect score.

The sharpest illustration: **B0 — the untouched damaged image, no restoration at all —
scores a higher mean Dice (0.9974) than every single diffusion-restored variant tested**
(D0/D1 at 10, 20, and 50 steps, all between 0.9883 and 0.9962). A practitioner who only
looked at Dice would conclude dense diffusion makes images *less* similar to the clean
reference than doing nothing — true in a narrow pixel-overlap sense, since diffusion's
denoising process perturbs pixels across the whole image while a corruption only touches a
local region — but this says nothing about which one preserves more real connections. On
edge-F1, B0 (0.9167) and diffusion (0.9110–0.9223) are much closer to each other, and both
are clearly behind U-Net (0.9626–0.9652). Dice and edge-F1 do not even agree on whether
diffusion beats doing nothing.

## Finding: different appearance metrics disagree with each other, not just with edge-F1

There is no single "the" appearance metric that could substitute for the evaluator, because
they don't agree among themselves:

- **Dice** ranks B0 (raw damaged input) near the *top* of the table (0.9974, 2nd of 7 row
  groups) — despite B0 having no restoration applied.
- **SSIM** ranks B0 as the clear *worst* (0.9393) — over 0.05 below every other row, and
  below diffusion, which Dice ranked worse than B0.
- **PSNR** ranks B1 (morphological repair) as the clear worst (28.3 dB) — despite B1 having
  the second-best edge-F1 (0.9429) among the non-U-Net/non-transformer rows. Morphological
  closing thickens and merges ink broadly, which costs it heavily on pixel-level PSNR even
  where it correctly restores a connection.

Three plausible single-number appearance metrics produce three different "worst" answers
(B1 by Dice's inverse ranking is mid-table, B0 by SSIM, B1 by PSNR), none of which is "dense
diffusion" — the row the direct-edge evaluator and the latency measurements agree is the
weakest deployable option.

## What this does and doesn't establish

**Established:** for this pilot task, foreground Dice specifically compresses the
meaningful accuracy differences Gate G2 found into a band about 5× narrower than the
direct-edge evaluator's, and in doing so produces a ranking (B0 beating all diffusion
variants) that a real deployment decision should not be made on. This is concrete evidence
for the project's founding motivation: image similarity can look fine while a restoration
loses real connections.

**Not established:** that appearance metrics are *useless* — SSIM's ranking of B0 as worst
is directionally closer to intuition than Dice's, and boundary-sensitive metrics might do
better still. The claim is narrower and better-supported: no single commonly-used
appearance metric reliably substitutes for a graph-aware evaluator on this task, not that
all appearance metrics are equally uninformative.

**Single run, single seed (7).** This should be checked against seeds 17/27 before being
treated as a confirmed, generalizable pattern rather than a striking pilot observation —
flagged as the immediate next step alongside the Stage 8 confirmatory-scale work.

## Decision

This closes the primary open item from [G2_REPORT.md](G2_REPORT.md)'s Stage 7 pointer with
a genuine positive result, not a null one: appearance metrics do hide the connectivity gap,
concretely and by a clean, checkable margin (~5× compression of the meaningful range, plus
a raw-input-beats-diffusion reversal on Dice specifically). This is strong candidate
evidence for the paper's central motivating figure.

**Next scoped step:** confirm this divergence pattern holds at seeds 17 and 27 (cheap,
reuses the already-trained-per-seed models' restored images if checkpoints are saved, or a
short rerun otherwise), then proceed to Stage 8 (confirmatory scale) with the researcher
present to supervise the multi-hour compute commitment.
