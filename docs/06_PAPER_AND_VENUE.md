# Paper and venue plan

Prepared 12 September 2026; revised 13 September 2026 after Gate G2 triggered the scope amendment
in [docs/01_RESEARCH_SCOPE.md](01_RESEARCH_SCOPE.md). Venue fit is a recommendation; acceptance
depends on original, convincing results and peer review.

## What paper this is now

Working title: **A Direct-Edge Benchmark for Diagram Connectivity Restoration**.

The originally intended paper (below, for the record) would have shown connection-specific
routing beating dense diffusion, fewer-step sampling, and generic adaptive routing on the
direct-edge accuracy/runtime frontier. Gate G2 showed the premise does not hold at this pilot
scale: a plain U-Net beats dense diffusion on both accuracy and latency, confirmed across three
seeds and four independent debugging interventions (more training, EMA + LR schedule, more
capacity — all tested and eliminated as fixable defects; see
[results/milestone2/G2_REPORT.md](../results/milestone2/G2_REPORT.md)). Per this plan's own
failure-response rule, the routing method is dropped rather than pursued anyway.

The paper this project can now honestly support: a reproducible diagram-corruption benchmark and
direct-edge evaluator, applied to a rigorous, confound-controlled comparison of restoration
architectures, that demonstrates (1) image-similarity metrics can hide connectivity failures a
graph-aware evaluator catches, and (2) for this task and scale, simple convolutional restoration
dominates diffusion-based restoration on both quality and speed — a useful negative result for
anyone considering diffusion for structural/graph-preserving restoration tasks. This still needs a
distinct, evidence-backed contribution (the generator + evaluator + benchmark, not just "diffusion
didn't work here") and the two pieces of evidence not yet gathered (Stages 7 and 9 below) before
it is submission-ready.

*Original framing, superseded:* "The strongest intended paper would show that connection-specific
routing provides a better direct-edge accuracy/runtime tradeoff than dense diffusion, fewer-step
sampling, generic adaptive routing, and a generic router trained with the same structural
restoration loss." Kept for the record per this project's amendment discipline.

## Claim-to-evidence checklist (revised)

| Claim | Required evidence | Status |
| --- | --- | --- |
| Connection failures matter in this task | Deliberate evaluator fixtures; damaged/dense/fewer-step edge errors | **Done** — 80/80 fixtures pass; B0 damaged-input F1 = 0.917 vs. clean-render F1 = 1.0 |
| Image similarity can hide connectivity failures | At least one case where an appearance metric and edge-F1 disagree | **Open — Stage 7**, the next scoped experiment |
| Simple restoration dominates diffusion here | Same-split, same-seed comparison with latency, repeated across seeds | **Done** — U0/U1 vs. D0/D1, 3 seeds, `results/milestone2/G2_REPORT.md` |
| The dominance isn't a fixable diffusion defect | Undertraining, training-trick, and capacity checks | **Done** — all three tested and ruled out |
| The result generalizes | Fresh held-out parents/layouts, corruption shifts, and (resources permitting) independently created examples | **Open — Stage 9** |
| Diffusion has *any* useful role | Honest accounting of where it's competitive, if anywhere, across the full sampling-step sweep | **Partially done** — 10/20/50 steps tested, none competitive; 1–5 steps untested |

Write every result sentence from saved metrics. Use placeholders for unavailable measurements in internal drafts; never populate tables with expected gains.

## Manuscript structure (revised)

1. **Introduction:** the diagram connection failure, why image similarity can miss it, and the exact contribution supported by the data — now a benchmark and comparative finding, not a routing method.
2. **Related work:** conditional restoration and diffusion-based restoration (Palette, DDIM), U-Net baselines, topology-sensitive supervision (clDice) — reposition against diagram/graphics-recognition **benchmark and evaluation** papers, not efficient-diffusion/routing papers (DyDiT etc. move to a shorter "why we did not pursue routing" note). **Needs the Stage 12 literature refresh** before drafting — the current [docs/02_LITERATURE_REVIEW.md](02_LITERATURE_REVIEW.md) was scoped for the dropped routing angle.
3. **Task and evaluation:** graph convention, corruptions, split discipline, direct-edge extraction, assisted-node limitation, and evaluation audits. *(unchanged — this is the generator/evaluator contribution, unaffected by the pivot.)*
4. **Method:** the compared architectures (morphological repair, U-Net with/without structural loss, deterministic transformer, conditional diffusion with/without structural loss) and training recipe, including the debugging interventions (EMA, LR schedule, capacity) as part of giving diffusion a fair chance.
5. **Experiments:** baselines first; the appearance-vs-edge divergence analysis (Stage 7); the 3-seed comparison with confidence intervals; the diffusion debugging ablations; confirmatory-scale results (Stage 8); generalization (Stage 9).
6. **Limitations:** synthetic bias, node-geometry assistance, ambiguous corruption, hardware dependence, and the fact that this compares one specific compact DiT configuration, not diffusion restoration in general.
7. **Conclusion:** a precise result supported by the experiment, without expanding beyond the tested diagram domain or overclaiming that diffusion never helps diagram restoration.

Required visuals: task examples with edge overlays; a figure contrasting an appearance-similar/
edge-different pair (Stage 7's core evidence); the accuracy/latency scatter across all seven
architectures and three seeds; the diffusion-debugging ablation table (steps × EMA/schedule ×
capacity); and failure panels. Use raw points and uncertainty where appropriate.

Artifact package: environment lock; dataset/renderer versions and licensing; split manifests and seeds; executable generation/training/evaluation entrypoints; exact configs; selected checkpoints where distributable; compact per-example metrics; timing procedure; evaluator fixtures; and one documented reproduction path. A third party should be able to reproduce at least the pilot and its primary table.

## Currently verified conference candidate

**ICDAR 2027** remains a strong fit — arguably a *better* fit than the original routing framing,
since its official call explicitly includes graphics recognition, document image processing, and
**gold-standard benchmarks/datasets**, which is now the paper's primary contribution type. Fit is
strongest if the completed results address diagram analysis and the evaluation is credible beyond
one generator (Stage 9). [Official call for papers](https://icdar2027.org/call-for-papers)

As posted on the official site and checked 12 September 2026:

| Item | Posted information |
| --- | --- |
| Abstract deadline | 31 January 2027, 23:59 AoE |
| Full paper deadline | 20 February 2027, 23:59 AoE |
| Paper format | Springer LNCS, maximum 17 pages including references and figures |
| Review | Double blind |
| Conference | 18–22 August 2027, Kuala Lumpur, Malaysia |

Recheck dates, author instructions, anonymity, artifact links, and authorship rules on the official site before preparing the final submission. These dates are not a commitment to submit and no registration/submission has been performed. [Official instructions and dates](https://icdar2027.org/call-for-papers)

For a narrower early result, consider an appropriately scoped graphics-recognition or document-analysis workshop once a current call is announced and verified. No specific open workshop or deadline is asserted here. Avoid forcing a weak result into the main conference solely because the date fits.

## Mentor review and submission readiness

Ask a faculty mentor to assess: whether the benchmark/negative-result framing is convincing without the routing method, the validity of the direct-edge evaluator, baseline strength, the statistical support for the U-Net-dominates finding (3 seeds), and venue fit. **Flag the scope pivot explicitly** — a mentor who was expecting the routing method should see the G2 evidence, not just the new framing. The current project is document-image research; if the academic requirement is specifically text mining, establish acceptability before extending implementation into an unrelated NLP component.

The final submission decision follows an evidence review: no hidden test tuning; no privileged inference inputs; no favorable-seed selection; all numerical claims traceable; limits stated; data/code rights checked; all authors have reviewed the final manuscript. This planning task prepares the route to that review and does not contact a mentor or submit externally.
