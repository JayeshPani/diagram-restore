# Research search log

Date: 12 September 2026. Input: the user's supplied `pasted-text.txt` brief, read in full. Domain: black-and-white diagram restoration; proposed method: connectivity-aware diffusion routing; environment: one M4 Pro Mac. No deadline or fixed venue was supplied.

## Sources and coverage

Consensus paper discovery and fetched records; arXiv abstracts and full-text HTML; ICLR proceedings; CVF results; official author/project repositories; official PyTorch documentation; official conference sites. No geographic filter was used because this is a methods study. The Consensus query used a 2020–2026 filter; older foundational works were retained when directly relevant. Recent versions through the search date were checked where discovered.

## Exact discovery queries

Consensus:

```text
connectivity topology preserving diagram image restoration adaptive diffusion token routing year:2020-2026
```

Web — method/domain families:

```text
connectivity aware dynamic routing diffusion diagram restoration topology
topology preserving image restoration diffusion clDice diagram line drawing
adaptive computation image restoration token routing efficient restoration transformer
"topology" "diagram" "restoration" diffusion
"Region-Adaptive Sampling" diffusion
"clDice" topology CVPR 2021
"Denoising" "line drawings" restoration topology
"diagram" "restoration" "connectivity" deep learning
"flowchart" "dataset" "connectors" recognition official
"ARF: Arbitrary Routing Framework" doi
"Dynamic Diffusion Transformer" ICLR 2025 github
```

Web — venue feasibility:

```text
ICDAR 2027 official conference document analysis recognition
ICPR 2026 official conference paper deadline conference dates
```

Direct fetches covered the three supplied arXiv URLs, related papers/versions discovered through results and author repositories, MPS documentation, and the ICDAR 2027 call for papers. In-paper searches for `Spatial-wise Dynamic Token`, `router`, and `Training` located method and resource details. Searches for `clDice` in the DyDiT HTML and `topology` in DC-DiT HTML returned no match; absence of a keyword is **not** evidence of novelty.

## Consensus record handling

The initial query returned ten records. Fetched before use:

- ARF: record `f079be962ab25c019763577180372849`.
- TRaM-VSR: record `0618be65db3a53a1b2e210158ff798fa`, then cross-checked against arXiv:2607.22231.
- PixRestore: record `7d296b63a29d54e8a8127a14e3d36501`, then cross-checked against arXiv:2608.16793.

Primary arXiv metadata takes precedence over abbreviated/inconsistent author names in the discovery records. Unfetched Consensus hits are not cited as evidence.

## Access and version notes

- Palette, DyDiT, and DC-DiT: inspected relevant full-text HTML sections. DyDiT's ICLR 2025 status was verified in conference proceedings.
- DyDiT++: latest discovered v4 title and author-reported TPAMI acceptance checked; publisher bibliographic details remain unverified.
- TRaM-VSR and PixRestore: arXiv abstracts and metadata verified; HTML accessible, but this planning comparison deliberately limits their detailed claims to abstract evidence.
- clDice: arXiv abstract and author repository inspected. The CVF landing-page fetch failed; CVPR status is also identified by the author repository. No new theorem is inferred from the abstract.
- ARF: Consensus record fetched; DOI discovered in an indexed record. Publisher DOI fetch failed and PubMed returned a browser check. Treat this as abstract-level evidence, not a full-paper verification.
- RAS: official project site and `microsoft/RAS` inspected; an initially attempted longer repository name failed and is not treated as a separate project.
- PyTorch “stable” documentation redirected to version 2.14 during the search. The local package is 2.9.0; implementation must verify APIs against its locked version.
- The DyDiT family is reconciled as related versions; the newer Alibaba repository and original NUS repository are implementation references, not two independent experiments.

## Exclusions and unresolved leads

Excluded networking routing/topology repair, power-system restoration, geological restoration, and surface-rendering line drawings because their task is not diagram image restoration. Reddit, generic paper summaries, and third-party “papers with code” pages were used at most for discovery, not technical conclusions. Conference-name collisions were excluded in favor of the intended document-analysis/pattern-recognition organizations' sites.

Mural restoration, D²iT, ART, task-adaptive medical restoration, and flowchart-recognition datasets are adjacent leads. They were not promoted to detailed evidence in this initial review because exact direct-edge restoration/routing relevance or usable annotations/resources require further inspection. A public flowchart detector dataset is not automatically a licensed paired restoration benchmark.

## Limits and next update

The major concept families were covered, but recent searches were still revealing relevant work. This is therefore a bounded initial review, **not search saturation**. It supports a pilot and identifies novelty risks; it does not certify originality or exhaust the 2026 literature.

Before committing to large experiments: inspect the full recent methods, broaden graphics-recognition/diagram reconstruction coverage, inspect citations to the closest family, and verify implementation licenses at fixed revisions. Before submission: repeat the core queries, update arXiv versions, recheck venue dates/rules, and revise the contribution if new work overlaps.

## Scope pivot (13 September 2026) — this search log is now stale for the paper framing

Gate G2 (`results/milestone2/G2_REPORT.md`) triggered the routing-to-benchmark amendment in
[docs/01_RESEARCH_SCOPE.md](01_RESEARCH_SCOPE.md). Every query and source above was directed at
positioning a **routing method** against DyDiT/DC-DiT/efficient-diffusion literature; none of it
searched for diagram/graphics-recognition **benchmark and evaluation** papers, which is the
project's actual remaining novelty question. This log has not been refreshed for that framing —
that refresh is revised Stage 12 in `docs/03_PROJECT_PLAN.md` and is a **required step before
drafting related work**, not optional. Suggested starting queries for that refresh: "diagram
restoration benchmark dataset", "graph-aware evaluation image restoration", "flowchart / circuit
diagram dataset direct edge annotation", "diffusion vs CNN restoration structural task comparison".

## First refresh pass (13 September 2026) — web search only, not full-text inspection

Queries run (web search, no domain filter, current-year results biased by the search tool):

```text
diagram restoration benchmark dataset graphics recognition 2026
graph-aware evaluation metric image restoration structural connectivity
flowchart circuit diagram dataset direct edge annotation restoration corruption
diffusion model versus CNN U-Net image restoration structural preservation comparison
"direct edge" OR "edge-level F1" graph extraction evaluation diagram image restoration 2025 2026
```

Findings and their disposition are in
[docs/02_LITERATURE_REVIEW.md](02_LITERATURE_REVIEW.md)'s "Literature refresh" section, most
notably **SciFlow-Bench** (arXiv:2602.09809) — a near-concurrent, independent argument for
"structural recoverability over visual similarity" in scientific-diagram *generation*, the
closest prior-art match to this project's Finding 04, and **DiagramNet** (arXiv:2605.01338), a
diagram+connection-annotation benchmark for *recognition* rather than restoration.

This pass is **evidence label B** (search-tool summaries and abstracts only; no full-text
inspection, no citation-graph check, no license verification) and is explicitly a first look,
not the required pre-submission pass. Every hit found was from 2026 or very late 2025 — this
area is moving fast enough that the queries should be repeated close to the submission date
rather than trusted from this single pass.

## Second refresh pass (14 September 2026) — full-text follow-up

Direct fetches and additional queries closing out the first pass's follow-up list:

```text
[WebFetch] https://arxiv.org/abs/2602.09809  (SciFlow-Bench, abstract page only — no venue/full method)
[WebFetch] https://arxiv.org/html/2602.09809  (SciFlow-Bench, full HTML — evaluator mechanism, authors, Table 4)
"Occluded Pages Restoration Benchmark" OR "OPRB" document restoration dataset degraded pages figures diagrams
diffusion model versus CNN structural fidelity diagram flowchart restoration graph accuracy comparison
circuit diagram wiring diagram restoration deep learning connectivity preservation benchmark
```

Outcomes, full detail in [docs/02_LITERATURE_REVIEW.md](02_LITERATURE_REVIEW.md):

- **SciFlow-Bench full-text read, done.** Its inverse parser is a hierarchical multi-agent
  VLM/OCR pipeline (not a single trained model, not an algorithmic geometry parser like this
  project's), using semantic sentence-embedding node matching over directed graphs. Purely
  an arXiv preprint (v3, June 2026), no venue found. It independently reports diffusion
  models (SDXL, PixArt-Σ) showing weak structural recoverability versus autoregressive VLMs
  — a different task and a different "winner" than this project's, but the same qualitative
  pattern (diffusion trailing on structural correctness), worth citing as adjacent
  corroborating evidence.
- **OPRB venue/authors verified.** It is introduced within **DocRevive** (Purkayastha et
  al.), accepted at the **CVPR 2026 Workshop on Multimodal Understanding for Long-form
  Analysis (MULA)**, arXiv:2604.10077. Document occlusion restoration (ink/stamps/scribbles
  removed), diagrams a ~23% minority; no graph/connectivity evaluation. OPRB's own
  restoration-evaluation methodology has not been read in full text.
- **Diagram-domain diffusion-vs-CNN structural comparison: still not found** after two
  search passes. Generic (non-diagram) diffusion literature argues the *opposite* direction
  from this project's finding — diffusion models are generally believed to produce *better*
  structural fidelity than CNNs in image generation broadly — making this project's result a
  genuine counter-to-conventional-wisdom finding worth foregrounding, not a confirmation of
  known results.
- **New find**: "Vision-Based Topology-Consistent Structural Parsing of Hand-Drawn Circuit
  Diagrams," *Sensors* 26(11):3440, 2026 (peer-reviewed, PMC13259090) — a real, non-synthetic
  1,317-diagram hand-drawn circuit benchmark with explicit connectivity reasoning. Parsing,
  not restoration; no architecture-family comparison. Relevant precedent that real hand-drawn
  diagram benchmarks are a tractable, publishable category — cited in docs/02 as motivation
  for this project's still-open renderer/acquisition-shift generalization item.

Remaining before the required pre-submission pass: OPRB's restoration methodology in full
text, SciFlow-Bench's Table 4 read directly (not via search-tool synthesis), and a proper
citation-graph check (who cites/is cited by SciFlow-Bench and DiagramNet) — none of this was
systematic literature-database work, only search-engine discovery across two passes.
