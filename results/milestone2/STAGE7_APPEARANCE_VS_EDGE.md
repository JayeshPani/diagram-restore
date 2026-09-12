# Stage 7 — Appearance metrics vs. the direct-edge evaluator

Prepared 13 September 2026, confirmed across seeds 7/17/27 the same day. Validation split,
same runs as [G2_REPORT.md](G2_REPORT.md) (unet_steps=6000, dit_steps=30000), now also
scoring mean foreground Dice, boundary F1, SSIM, and PSNR
(`evaluation.aggregate_image_metrics`) alongside edge-F1 for every row. Full numeric
output: [baselines.json](baselines.json), [seed17/baselines.json](seed17/baselines.json),
[seed27/baselines.json](seed27/baselines.json).

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

**Established, confirmed across 3 seeds:** for this pilot task, foreground Dice specifically
compresses the meaningful accuracy differences Gate G2 found into a band about 4.5× narrower
than the direct-edge evaluator's, and in doing so produces a ranking (B0 beating all
diffusion variants, 18/18 configs across seeds) that a real deployment decision should not
be made on. This is concrete evidence for the project's founding motivation: image
similarity can look fine while a restoration loses real connections.

**Not established:** that appearance metrics are *useless* — SSIM's ranking of B0 as worst
is directionally closer to intuition than Dice's, and boundary-sensitive metrics might do
better still. The claim is narrower and better-supported: no single commonly-used
appearance metric reliably substitutes for a graph-aware evaluator on this task, not that
all appearance metrics are equally uninformative.

## Confirmed across seeds 7/17/27

| Seed | Dice range | Edge-F1 range | Ratio (F1 range ÷ Dice range) | B0 beats every diffusion variant on Dice? |
| --- | --- | --- | --- | --- |
| 7 | 0.0111 | 0.0542 | 4.88× | Yes (6/6 diffusion configs) |
| 17 | 0.0113 | 0.0536 | 4.76× | Yes (6/6) |
| 27 | 0.0162 | 0.0612 | 3.79× | Yes (6/6) |
| **mean ± sd** | — | — | **4.48× ± 0.60** | **Yes, 18/18 diffusion configs across all seeds** |

Both headline patterns replicate at every seed tested: Dice compresses the meaningful range
by roughly 4–5×, and the untouched damaged input scores a higher mean Dice than every one
of the six diffusion sampling-step/loss configurations, at all three seeds — 18 out of 18
diffusion configurations checked. This is no longer a single-seed curiosity.

## Decision

This closes the primary open item from [G2_REPORT.md](G2_REPORT.md)'s Stage 7 pointer with
a genuine, now multi-seed-confirmed positive result: appearance metrics do hide the
connectivity gap, concretely and by a clean, checkable margin (~4.5× compression of the
meaningful range, plus a raw-input-beats-diffusion reversal on Dice specifically, at every
seed tested). This is strong evidence for the paper's central motivating figure.

**Next scoped step:** proceed to Stage 8 (confirmatory scale, 5,000/500/1,000 diagrams) with
the researcher present to supervise the multi-hour compute commitment — the two things that
would otherwise gate that decision (does the U-Net-dominates finding hold across seeds; does
the appearance-vs-edge divergence hold across seeds) are both now answered yes.
