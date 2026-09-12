# Staged project plan

Prepared 12 September 2026. The sequence follows all twelve stages of the supplied brief. The first substantive implementation result is the dataset plus evaluator.

## Working assumptions

One researcher, approximately 12–15 focused hours a week, access to the inspected M4 Pro Mac, and no assumed cloud GPU budget. Plan for ten working weeks plus revision buffer. This is an effort allocation, not a measured training-time forecast or a guarantee of completion. Re-estimate after the benchmark. Mentor availability, academic requirements, and venue choice remain open planning assumptions.

Proceed within a stage until its output is reviewable. At each scientific gate, record the result, limitations, and decision in a short report. A failed gate calls for investigation or a narrower claim; it does not justify silently changing the experiment to favor the method.

## Twelve stages and their checks

| Stage | Work | Concrete result to inspect | Completion or decision rule |
| --- | --- | --- | --- |
| 1. Exact scope | Define input/output, direct-edge semantics, exclusions, and primary hypothesis. | `docs/01_RESEARCH_SCOPE.md` | Complete for planning. Numerical targets are provisional until protocol freeze. |
| 2. Closest work | Compare the three named papers, their updates, topology losses, restoration routing, and recent efficient diffusion restoration. | `docs/02_LITERATURE_REVIEW.md` and search log | Initial review complete. Before method freeze, inspect the closest recent full methods and update the novelty statement. |
| 3. Reproducibility and budget | Create a Python 3.12 environment; lock dependencies; record device, Git revision, and seed controls. Run a short training and sparse-block benchmark. | Environment lock, `results/environment.json`, `results/benchmark_pilot.json` | Clean process can import packages and use MPS; measured costs fit the stage budget. Current inventory is complete; setup and benchmark are pending. |
| 4. Pilot data | Sample clean graphs, split parent diagrams, render, corrupt, and retain exact annotations. | Generator, manifest, 700 pairs, leakage audit, 36-example contact sheet | Re-running a fixed seed reproduces data; no parent/geometry leakage; images and graph labels pass geometric checks. |
| 5. Evaluator | Extract direct edges using known node regions but no ground-truth edge list inside extraction. Test deliberate errors. | Evaluation CLI, at least 40 hand-designed fixtures, annotated overlays and audit report | Every fixture has the expected edge set; clean rendered graphs yield exact recovery; prediction audits distinguish extraction errors from model errors. **Gate G1.** |
| 6. Baselines | Learn a tiny subset, then train simple repair/U-Net and compact dense diffusion. Include a deterministic transformer control. Vary sampling steps. | Saved configs/checkpoints; validation edge metrics, failures, and runtime curves | Pipeline learns the tiny set; a useful diffusion regime is demonstrated; reduced compute produces measurable connection loss or another clear tradeoff. Otherwise investigate. **Gate G2.** |
| 7. Oracle diagnostic | Use the same mask-tolerant backbone to compare equal-budget random, all-foreground, observation-edge, and ground-truth connector policies. | Budget-matched oracle table and routing overlays | Connector protection must improve edge recovery over random and clarify whether it adds more than generic foreground protection. If not, reconsider the criterion. **Gate G3.** |
| 8. Minimal learned router | Add feature/timestep router, connector-priority supervision, structural restoration loss, and compute regularization. Implement gather/MLP/scatter inference. | One adaptive model, sparse/dense-mask parity check, gate-gradient check, actual token counts, routing maps | Model takes only damaged pixels at inference; saved counters prove skipped operations; training is stable. No quality claim yet. |
| 9. Attribution and ablations | Compare dense, fewer-step, generic, loss-only, foreground routing, and proposed variants at equal tuning effort. | Validation-selected accuracy/runtime frontier and ablation table | Connection-specific supervision improves on generic routing with the same restoration loss. If structural loss alone explains the gain, revise the contribution. **Gate G4.** |
| 10. Real speed and generalization | Freeze choices; expand data only after pilot gates; repeat key systems across seeds; evaluate held-out layouts, line widths, gaps, and independent rendering/scan examples. | Test report, timing repetitions, uncertainty estimates, OOD breakdown, extraction audit | Primary success condition is supported, generalization is characterized, and overhead is included. A synthetic-only result must be labeled as such. **Gate G5.** |
| 11. Evidence-based claim | Write the strongest claim actually supported, including cases where simpler models win. | One-paragraph contribution statement and claim-to-evidence table | Every numerical claim traces to a saved run; unsupported speed, topology-guarantee, or generalization language is removed. |
| 12. Paper and submission preparation | Produce figures/tables, manuscript, limitations, artifact instructions, and mentor review. Recheck venue rules. | Complete manuscript, reproducibility package, venue checklist | Evidence and venue scope align; complete author review before submission. No submission is performed by this planning task. |

## Ten-week working schedule

| Week | Main work | Reviewable result |
| --- | --- | --- |
| 1 | Scope, literature, environment, generator v1 | Scope and literature files; environment lock; 700-pair dataset and contact sheet |
| 2 | Evaluator fixtures, leakage checks, training benchmark | G1 report and measured experiment budget |
| 3 | Tiny-set learning, simple restoration and dense baselines | Baseline learning curves and sanity report |
| 4 | Sampling-budget sweep and failure analysis | G2 decision with image/graph examples |
| 5 | Oracle and sparse-operation feasibility | G3 decision and block/full-model timing breakdown |
| 6 | Learned router and first ablations | Adaptive checkpoint, routing maps, sparse equivalence report |
| 7 | Fair controls and validation selection | G4 decision; frozen metrics, thresholds, settings and seeds |
| 8 | Larger split, repeated training, held-out stress sets | Independent test measurements and G5 evidence |
| 9 | Independent examples, failure audit, tables and draft | Complete results section and manuscript draft |
| 10 | Mentor review, novelty refresh, artifact checks | Reviewable conference submission package |

If starting on 14 September 2026, ten calendar weeks end around 22 November. This leaves revision time before the candidate venue's currently posted 2027 deadlines. Longer measured training or failed scientific gates can extend the schedule.

## First milestone: task checklist

- [ ] Build graph sampler for 2–5 labeled nodes, boxes/circles, and unambiguous undirected direct edges.
- [ ] Define shape masks, boundary attachment rules, line rasterization, and minimum clearances.
- [ ] Generate parent IDs and split 500/100/100 before any corruption or augmentation.
- [ ] Save clean/damaged images, node geometry, true edge list, connector masks, seeds, and corruption metadata.
- [ ] Produce training/validation contact sheets covering clean, noise, blur, and line-gap cases.
- [ ] Implement edge extraction from prediction and node geometry only.
- [ ] Add fixtures for intact links, missing links, invented links, wrong endpoints, near-touches, and indirect paths.
- [ ] Check clean-render graph recovery, manifest uniqueness, reproducibility, and annotation/image agreement.
- [ ] Audit at least 30 development examples and every failing fixture.
- [ ] Save a concise G1 report with examples and failure counts.

## How to respond to a failed gate

| Finding | Response within the evidence |
| --- | --- |
| Evaluator confuses node outlines with connectors | Fix graph semantics/extraction before measuring models. |
| Most corruptions are trivial or fundamentally ambiguous | Revise the development distribution once with documented criteria; freeze a fresh held-out split after development. |
| U-Net dominates the useful quality/runtime range | Stop expanding the diffusion method; assess a narrower restoration/evaluation study or a justified new setting. |
| Oracle gives no benefit after a fair mask-tolerant control | Reconsider connectivity-specific allocation; do not build a complex router to rescue a failed premise. |
| Structural loss explains all improvements | Credit structural supervision; do not attribute the gain to routing. |
| Sparse execution is slower | Report operation savings only; test the already planned 128-pixel scale if justified. Further kernels/hardware require a revised resource plan. |
| Only the pilot generator shows gains | Limit the claim to that domain and strengthen independent validation before a broader paper. |
| A close paper covers the proposed signal | Reposition around a demonstrably different question or benchmark, or revise the project before expensive training. |

## Stage report template

For each gate record: question; fixed configuration; data split/version; run IDs; observations; representative successes/failures; uncertainty; compute consumed; decision; next scoped step. Use “planned,” “measured,” and “inferred” explicitly. Preserve failed runs and any protocol amendments.
