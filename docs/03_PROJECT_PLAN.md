# Staged project plan

Prepared 12 September 2026; revised 13 September 2026 after the Stage 6 baseline result triggered
this plan's own failure-response rule (row 3 of the table below). The first six stages are
complete and unchanged; stages 7–12 are rescoped from a routing method to a benchmark
contribution. See [docs/01_RESEARCH_SCOPE.md](01_RESEARCH_SCOPE.md)'s amendment and
[results/milestone2/G2_REPORT.md](../results/milestone2/G2_REPORT.md) for the evidence.

## Working assumptions

One researcher, approximately 12–15 focused hours a week, access to the inspected M4 Pro Mac, and no assumed cloud GPU budget. Plan for ten working weeks plus revision buffer. This is an effort allocation, not a measured training-time forecast or a guarantee of completion. Re-estimate after the benchmark. Mentor availability, academic requirements, and venue choice remain open planning assumptions.

Proceed within a stage until its output is reviewable. At each scientific gate, record the result, limitations, and decision in a short report. A failed gate calls for investigation or a narrower claim; it does not justify silently changing the experiment to favor the method.

## Stages 1–6: complete, unchanged

| Stage | Work | Concrete result to inspect | Completion or decision rule |
| --- | --- | --- | --- |
| 1. Exact scope | Define input/output, direct-edge semantics, exclusions, and primary hypothesis. | `docs/01_RESEARCH_SCOPE.md` | Complete; revised in the v0.2 amendment above. |
| 2. Closest work | Compare the three named papers, their updates, topology losses, restoration routing, and recent efficient diffusion restoration. | `docs/02_LITERATURE_REVIEW.md` and search log | Initial review complete for the routing question. **Needs a refresh** for the benchmark framing (comparable diagram/graphics-recognition benchmark papers) before submission — not yet done. |
| 3. Reproducibility and budget | Create a Python 3.12 environment; lock dependencies; record device, Git revision, and seed controls. Run a short training and sparse-block benchmark. | Environment lock, `results/milestone1/environment.json`, `results/milestone1/benchmark_pilot.json` | Complete. |
| 4. Pilot data | Sample clean graphs, split parent diagrams, render, corrupt, and retain exact annotations. | Generator, manifest, 700 pairs, leakage audit, contact sheets | Complete: fixed-seed reproduction, no parent/geometry leakage, 700/700 records pass integrity checks. |
| 5. Evaluator | Extract direct edges using known node regions but no ground-truth edge list inside extraction. Test deliberate errors. | Evaluation CLI, 80 hand-designed fixtures, annotated overlays and audit report | **Gate G1 passed**: 80/80 fixtures pass; clean-render recovery is exact (F1 = 1.0, exact-graph = 1.0 on 700 images). |
| 6. Baselines | Learn a tiny subset, then train simple repair/U-Net and compact dense diffusion. Include a deterministic transformer control. Vary sampling steps. | Saved metrics and runtime curves in `results/milestone2/` | **Gate G2 screened**: U0/U1 dominate D0/D1 on accuracy and latency, confirmed across 3 seeds and 4 independent debugging interventions (undertraining, training tricks, capacity, seed variance all ruled out as explanations). This **triggered the "U-Net dominates" failure-response row** below — see `results/milestone2/G2_REPORT.md`. |

## Stages 7–12 (revised): a benchmark contribution, not a routing method

Stage 6's result removes the premise for Stages 7–9 as originally written (oracle diagnostic,
learned router, routing ablations — all conditional on diffusion being worth accelerating). Per
the failure-response table's own instruction ("do not build a complex router to rescue a failed
premise"), those stages are dropped rather than attempted. The remaining work turns the
already-validated generator, evaluator, and comparison into a complete benchmark paper.

| Stage | Work | Concrete result to inspect | Completion or decision rule |
| --- | --- | --- | --- |
| 7. Appearance-vs-edge divergence | Compute image-similarity metrics (Dice/PSNR/SSIM) for every Stage 6 model alongside their edge-F1, on the same validation images. | `results/milestone2/STAGE7_APPEARANCE_VS_EDGE.md` | **Done, confirmed at 3 seeds.** Mean Dice range is 4.48× ± 0.60 narrower than edge-F1's range across seeds 7/17/27; Dice ranks the untouched damaged input above every diffusion-restored variant tested (18/18 configs). SSIM and PSNR disagree with Dice and each other about which row is worst. |
| 8. Expanded confirmatory benchmark | Expand to the 5,000/500/1,000 split (per `docs/04_EXPERIMENT_PROTOCOL.md` §9); repeat the retained comparison (B0/B1/U0/U1/T0/D0/D1) at 3 seeds on the larger split. | `results/milestone3/STAGE8_CONFIRMATORY_REPORT.md` | **Done.** Both findings replicate on a freshly generated dataset and get *stronger*: the U0/U1-vs-D0/D1 gap widens from ~0.05 to ~0.069 mean F1 (gap-to-noise ratio ~7-9x → ~29x), and the Dice/edge-F1 divergence widens from ~4.5x to ~6-7.5x compression. Not yet done: additional baseline architectures (second U-Net width, plain CNN autoencoder) a reviewer might expect — deprioritized once the core replication was this decisive. |
| 9. Generalization | Evaluate held-out layout families, thinner lines, stronger corruptions, and — resources permitting — a second synthetic renderer or independently authored diagrams (per `docs/04_EXPERIMENT_PROTOCOL.md` §9). | OOD breakdown table, extraction-failure audit. | At least one genuine distribution shift is tested and reported honestly, including any case where the ranking changes. |
| 10. Evidence-based claim | Write the strongest claim the benchmark actually supports, including where diffusion is competitive (if any operating point exists) and where it clearly is not. | One-paragraph contribution statement and claim-to-evidence table (`docs/06_PAPER_AND_VENUE.md`). | Every numerical claim traces to a saved run; no routing/acceleration language remains. |
| 11. Paper and submission preparation | Produce figures/tables, manuscript, limitations, artifact instructions, and mentor review. Recheck venue rules. | Complete manuscript, reproducibility package, venue checklist. | Evidence and venue scope align; complete author review before submission. No submission is performed by this planning task. |
| 12. Literature refresh | Before submission, redo the closest-work search (Stage 2) specifically for diagram/graphics-recognition **benchmark and evaluation** papers, not routing/efficient-diffusion papers — the positioning target has changed. | Updated `docs/02_LITERATURE_REVIEW.md` and `docs/07_SEARCH_LOG.md`. | **First pass done** (web search only, evidence label B): closest match is SciFlow-Bench (arXiv:2602.09809, near-concurrent independent argument for structural-recoverability-over-visual-similarity in diagram *generation*). Full-text read and citation-graph check still needed before drafting. |

## Revised working schedule

The original ten-week schedule assumed Weeks 5–8 on the oracle/router/ablations. Weeks 1–4 are
complete as originally planned (through Gate G2); the remainder is replaced:

| Week | Main work | Reviewable result |
| --- | --- | --- |
| 1 (done) | Scope, literature, environment, generator v1 | Scope and literature files; environment lock; 700-pair dataset and contact sheet |
| 2 (done) | Evaluator fixtures, leakage checks, training benchmark | Gate G1 report |
| 3–4 (done) | Baselines, sampling-step sweep, 4-way confound debugging, 3-seed confirmation | Gate G2 screening report (`results/milestone2/G2_REPORT.md`) and this plan's revision |
| 5 (done) | Appearance-vs-edge divergence analysis (Stage 7), confirmed at 3 seeds | `results/milestone2/STAGE7_APPEARANCE_VS_EDGE.md` — the paper's core motivating evidence |
| 6–7 (done) | Expanded confirmatory benchmark at the larger split, 3 seeds (Stage 8) | `results/milestone3/STAGE8_CONFIRMATORY_REPORT.md` — both findings replicate and strengthen |
| 8 | Generalization: held-out shift, second renderer if feasible (Stage 9) | OOD breakdown and extraction audit |
| 9 (partial) | Literature refresh for benchmark framing (Stage 12, first pass done), draft manuscript (Stage 10) | Updated lit review; complete results section and manuscript draft |
| 10 | Mentor review, artifact checks, submission package (Stage 11) | Reviewable conference submission package |

If restarting the remaining schedule on 15 September 2026, six weeks end around 27 October — inside the candidate venue's currently posted 2027 deadlines with revision buffer to spare, since the routing stages that would have consumed Weeks 5–8 are no longer needed.

## First milestone: task checklist (complete)

- [x] Build graph sampler for 2–5 labeled nodes, boxes/circles, and unambiguous undirected direct edges.
- [x] Define shape masks, boundary attachment rules, line rasterization, and minimum clearances.
- [x] Generate parent IDs and split 500/100/100 before any corruption or augmentation.
- [x] Save clean/damaged images, node geometry, true edge list, connector masks, seeds, and corruption metadata.
- [x] Produce training/validation contact sheets covering clean, noise, blur, and line-gap cases.
- [x] Implement edge extraction from prediction and node geometry only.
- [x] Add fixtures for intact links, missing links, invented links, wrong endpoints, near-touches, and indirect paths.
- [x] Check clean-render graph recovery, manifest uniqueness, reproducibility, and annotation/image agreement.
- [x] Audit at least 30 development examples and every failing fixture.
- [x] Save a concise G1 report with examples and failure counts.

## How to respond to a failed gate

| Finding | Response within the evidence |
| --- | --- |
| Evaluator confuses node outlines with connectors | Fix graph semantics/extraction before measuring models. |
| Most corruptions are trivial or fundamentally ambiguous | Revise the development distribution once with documented criteria; freeze a fresh held-out split after development. |
| **U-Net dominates the useful quality/runtime range** | **Triggered 13 September 2026.** Stop expanding the diffusion method; assess a narrower restoration/evaluation study — done, see the revised Stages 7–12 above. |
| Oracle gives no benefit after a fair mask-tolerant control | Not reached — no oracle was built once the U-Net-dominates row triggered first. |
| Structural loss explains all improvements | Not applicable to the revised scope; structural loss remains one compared condition (U1/D1) in the benchmark. |
| Sparse execution is slower | Not applicable — no sparse execution was built. |
| Only the pilot generator shows gains | Limit the claim to that domain and strengthen independent validation before a broader paper — this is now Stage 9. |
| A close paper covers the proposed signal | Reposition around a demonstrably different question or benchmark, or revise the project before expensive training. |

## Stage report template

For each gate record: question; fixed configuration; data split/version; run IDs; observations; representative successes/failures; uncertainty; compute consumed; decision; next scoped step. Use “planned,” “measured,” and “inferred” explicitly. Preserve failed runs and any protocol amendments.
