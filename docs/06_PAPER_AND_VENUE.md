# Paper and venue plan

Prepared 12 September 2026. Venue fit is a recommendation; acceptance depends on original, convincing results and peer review.

## What paper this could become

Working title: **Connectivity-Aware Adaptive Diffusion for Diagram Restoration**.

The strongest intended paper would show that connection-specific routing provides a better direct-edge accuracy/runtime tradeoff than dense diffusion, fewer-step sampling, generic adaptive routing, and a generic router trained with the same structural restoration loss. It would also show where a conventional U-Net is preferable.

A methods paper needs evidence that the routing policy contributes more than an existing bypass architecture plus a known loss. If the method does not improve that frontier, a carefully validated diagram-restoration benchmark or failure analysis may be a more honest output, but that alternative also needs a distinct contribution and suitable venue. Do not treat a negative outcome or a synthetic dataset alone as guaranteed publication material.

## Claim-to-evidence checklist

| Claim | Required evidence |
| --- | --- |
| Connection failures matter in this task | Deliberate evaluator fixtures; damaged/dense/fewer-step edge errors; visual failures with similar appearance scores |
| Choosing connector regions matters | Same-backbone, same-budget oracle comparisons against random, foreground, and observation-edge policies |
| The learned policy adds value | C0 versus G1 and F0, with equal structural loss and fair training/tuning |
| The architecture executes less work | Active-token counters, sparse/dense-mask equivalence, and operation-cost accounting |
| Restoration is faster on the tested Mac | Synchronized end-to-end measurements including routing, repeated sessions, and reported uncertainty |
| The result generalizes | Fresh held-out parents/layouts, corruption shifts, multiple seeds, and independently created examples |
| Diffusion has a useful role | Honest quality/runtime comparison with U0/U1 and the deterministic transformer |

Write every result sentence from saved metrics. Use placeholders for unavailable measurements in internal drafts; never populate tables with expected gains.

## Manuscript structure

1. **Introduction:** the diagram connection failure, why image similarity can miss it, and the exact contribution supported by the data.
2. **Related work:** conditional restoration, dynamic diffusion, efficient restoration, and topology-sensitive supervision; position explicitly against the closest recent methods.
3. **Task and evaluation:** graph convention, corruptions, split discipline, direct-edge extraction, assisted-node limitation, and evaluation audits.
4. **Method:** dense conditional model, feature/timestep router, priority targets, objective, and actual sparse inference.
5. **Experiments:** simple baselines first; budget curves; oracle; matched controls and ablations; timings; seeds; distribution shifts.
6. **Limitations:** synthetic bias, node-geometry assistance, ambiguous corruption, hardware dependence, remaining failure cases, and cases favoring U-Net.
7. **Conclusion:** a precise result supported by the experiment, without expanding beyond the tested diagram domain.

Required visuals: task examples with edge overlays; architecture showing training-only labels versus inference inputs; connection-F1 versus latency frontier; oracle and ablation tables; early/middle/late routing maps; and failure panels. Use raw points and uncertainty where appropriate. A favorable routing heatmap is explanatory evidence, not a substitute for an ablation.

Artifact package: environment lock; dataset/renderer versions and licensing; split manifests and seeds; executable generation/training/evaluation entrypoints; exact configs; selected checkpoints where distributable; compact per-example metrics; timing procedure; evaluator fixtures; and one documented reproduction path. A third party should be able to reproduce at least the pilot and its primary table.

## Currently verified conference candidate

**ICDAR 2027** is a relevant candidate because its official call explicitly includes graphics recognition, document image processing, and gold-standard benchmarks/datasets. Fit is strongest if the completed results address diagram analysis and the evaluation is credible beyond one generator. The main track is conditional on the strength of the completed study. [Official call for papers](https://icdar2027.org/call-for-papers)

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

Ask a faculty mentor to assess the precise novelty relative to DyDiT/DyDiT++, the validity of the direct-edge evaluator, baseline strength, statistical support, and venue fit. The current project is document-image research; if the academic requirement is specifically text mining, establish acceptability before extending implementation into an unrelated NLP component.

The final submission decision follows an evidence review: no hidden test tuning; no privileged inference inputs; no favorable-seed selection; all numerical claims traceable; limits stated; data/code rights checked; all authors have reviewed the final manuscript. This planning task prepares the route to that review and does not contact a mentor or submit externally.
