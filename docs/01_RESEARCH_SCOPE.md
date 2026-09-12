# Research scope

Version 0.2 · 13 September 2026 · supersedes v0.1's routing hypothesis; see amendment below.

## Scope amendment (13 September 2026)

Version 0.1 proposed a connectivity-aware routing method to accelerate diffusion-based diagram
restoration. Stage 6 baseline experiments (three seeds, four independent debugging
interventions — more optimizer steps, EMA + a warmup-cosine learning-rate schedule, a 2.9× larger
model, and repeated seeds) showed that a plain small U-Net beats the compact conditional diffusion
transformer on this task by roughly 0.05 direct-edge F1 while running 10×–74× faster, and that
none of the four interventions closed the gap — two of them made it worse. Full evidence trail:
[results/milestone2/G2_REPORT.md](../results/milestone2/G2_REPORT.md).

Per this document's own "Practical relevance" clause below (unchanged from v0.1) and the staged
plan's failure-response rule, that result is the documented trigger to stop expanding the
diffusion/routing method rather than build a router to rescue a losing backbone. This version
rescopes the project around the parts of v0.1 that are unaffected and already validated: the
corruption generator, the direct-edge evaluator, and a rigorous, confound-controlled comparison of
restoration architectures. Sections below marked *(unchanged from v0.1)* still hold; everything
else is revised.

**Working title:** A Direct-Edge Benchmark for Diagram Connectivity Restoration.

**Research question:** Does image-similarity evaluation hide connectivity failures in diagram
restoration, and which restoration architectures actually preserve direct connections under a
controlled, reproducible corruption benchmark?

**Problem to establish.** *(unchanged from v0.1, now demonstrated)* A restored diagram can look
similar to its clean reference while losing a short connection or inventing a shortcut. Stage 7
confirmed this at a concrete, checkable margin: mean foreground Dice spans only 0.9875–0.9986
across all seven compared methods (a range of 0.0111) while edge-F1 spans 0.9110–0.9652 (a range
of 0.0542, ~5× wider) — and Dice ranks the untouched damaged input above every diffusion-restored
variant tested. Single seed; see
[results/milestone2/STAGE7_APPEARANCE_VS_EDGE.md](../results/milestone2/STAGE7_APPEARANCE_VS_EDGE.md).

**Input and output.** *(unchanged from v0.1)* The model receives only a damaged grayscale raster
image. The target is a clean black-and-white image with the original direct connections. The
initial domain contains boxes, circles, and undirected connecting lines at 64 × 64 pixels, with
mild noise, blur, and short line gaps. Graph annotations support training and evaluation; node
locations and true edges are never additional inference inputs. The pilot uses separated connector
paths with no ambiguous crossings or free-standing junctions.

**The benchmark contribution.** A reproducible synthetic diagram-corruption generator (seeded,
leakage-checked, strata-controlled) and a direct-edge evaluator that distinguishes a direct A–C
edge from the indirect path A–B–C — validated against 80 hand-specified counterexample fixtures —
paired with a rigorous comparative study: damaged input, morphological repair, a small U-Net
(with/without a structural loss), a deterministic transformer control, and dense conditional
diffusion (with/without the structural loss, across a 10/20/50-step sampling sweep, three training
seeds, and four independent checks that the diffusion result is not a fixable training artifact).
This replaces v0.1's routing method as the primary contribution; the generator and evaluator are
unaffected by the pivot and were already validated at Gate G1 before the routing result came in.

**Essential comparisons.** Damaged input (B0); simple morphological repair (B1); small U-Net,
with and without a structural loss (U0/U1); a deterministic transformer using the diffusion
backbone (T0); dense conditional diffusion, with and without the structural loss, across a
sampling-step sweep (D0/D1). *(v0.1's routing-specific rows — generic routing, foreground/edge
routing, the oracle policy, and the complete routing proposal — are dropped; see the amendment.)*

**Success criteria (revised).** A benchmark contribution is supported by: (1) validated fixtures
and clean-graph recovery for the evaluator (met — see
[results/milestone1](../results/milestone1)); (2) a multi-seed, confound-controlled comparison
showing which architectures dominate the accuracy/latency frontier for this task (met — see
[results/milestone2/G2_REPORT.md](../results/milestone2/G2_REPORT.md)); (3) at least one
demonstrated case where an appearance metric and the direct-edge F1 disagree on which restoration
is better (met, single seed — see
[results/milestone2/STAGE7_APPEARANCE_VS_EDGE.md](../results/milestone2/STAGE7_APPEARANCE_VS_EDGE.md));
(4) generalization evidence beyond the exact pilot generator: a held-out layout/corruption
distribution shift and, resources permitting, a second renderer or independently authored diagrams
(revised Stage 9, not yet gathered). Report edge precision/recall, invented and missing edges, and
variability across at least three training seeds (met for the primary comparison, not yet repeated
for item 3) with confidence intervals before any generalization claim.

**Practical relevance.** *(unchanged from v0.1 — this is the clause whose condition was met)*
U-Net remains an essential reference. If it matches or exceeds diffusion's connectivity accuracy
at lower latency throughout the useful operating range, report that outcome and reconsider the
diffusion contribution. **This condition has been met** (Stage 6, three seeds); the project is
rescoped accordingly rather than continuing to a routing method built on the losing backbone.

**Exclusions.** *(unchanged from v0.1)* No OCR, arrow-direction prediction, complex circuit
symbols, text generation, full-page document understanding, user interface, or large pretrained
image generator in the pilot. This project belongs to document image analysis/graphics
recognition. Any course requirement for a text-mining component needs a separate scope decision.

**First checkable result.** *(met, unchanged from v0.1)* A reproducible 500/100/100 clean-diagram
split, inspected clean/damaged examples, and a tested evaluator that distinguishes a direct A–C
edge from the indirect path A–B–C.
