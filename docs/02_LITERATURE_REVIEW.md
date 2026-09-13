# Literature and implementation review

Search date: 12 September 2026. This is a focused planning review, not an exhaustive systematic review or proof of novelty.

## Scope and main conclusion

The supplied brief proposes adaptive diffusion for restoring connections in simple diagrams. The relevant literature spans conditional restoration, dynamic diffusion computation, topology-sensitive learning, and efficient restoration. The first three named papers were checked directly; searches also covered subsequent work and implementation sources.

The project is worth a controlled pilot, but its architectural novelty is presently **unproven**. DyDiT already contains the proposed feedforward bypass with shared attention. DC-DiT already allocates tokens according to image content and denoising stage. Therefore, the question to investigate is whether **supervising the routing decision for direct-connection recovery** improves the measured connectivity/runtime frontier beyond generic routing and structural restoration losses. [DyDiT method](https://arxiv.org/html/2410.03456v2), [DC-DiT method](https://arxiv.org/html/2603.06351v2)

## Review method and evidence labels

Discovery used Consensus and web search. The three shortlisted Consensus records were fetched before use; recent diffusion papers were cross-checked on arXiv. Important method details for the three starting papers were inspected in full-text HTML. Publication status was checked against proceedings or the paper's own metadata where available. Author-provided arXiv acceptance notes are distinguished from publisher verification.

Evidence labels: **A** = relevant full-text method/experiment sections inspected; **B** = authoritative abstract/metadata or fetched paper record, with claims restricted accordingly; **C** = primary implementation documentation, not independent performance verification. None of the reported methods was reproduced during planning.

Rank is reading/implementation priority. Fit uses qualitative bands reflecting problem/outcome match (30%), method overlap (25%), setting/resources (20%), evidence (15%), and recency (10%); numerical scores would imply more precision than this initial review supports.

## Ranked paper comparison

| Rank / fit | Paper and status | Problem and method | Training requirements / supported result | Overlap, limitation, and project implication | Evidence |
| --- | --- | --- | --- | --- | --- |
| 1 / Very high | **Dynamic Diffusion Transformer**, Zhao et al., ICLR 2025, first preprint 2024. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a44a70acd5d0abc1a252ada9719dd06d-Abstract-Conference.html) | Dynamic width across timesteps and spatial token selection for image generation. | Fine-tunes pretrained DiT. Authors report 51% fewer FLOPs and 1.73× speedup for one DiT-XL setting. | Its SDT component already skips MLP updates while retaining attention, with gather/scatter inference. This is the closest architectural precedent, not a novel mechanism for us. Published scale/hardware/results do not establish benefits on small Mac-trained diagrams. [Method](https://arxiv.org/html/2410.03456v2) | A |
| 2 / Very high | **DC-DiT: Adaptive Compute and Elastic Inference for Visual Generation via Dynamic Chunking**, Haridas et al., 2026 preprint, v2. [Paper](https://arxiv.org/abs/2603.06351) | Learned encoder/router/decoder compresses locally predictable token regions; token budgets vary with content and denoising stage. | Diffusion training with compression regularization and multiple budgets; reported ImageNet improvements include up to 36.8% fewer inference FLOPs. | Adaptive importance and budget flexibility are already covered. It changes the token sequence processed by the backbone, unlike our proposed MLP-only adaptation. Training uses large accelerator resources and packed attention; it is a conceptual reference, not the pilot implementation. [Full text](https://arxiv.org/html/2603.06351v2) | A |
| 3 / High | **clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation**, Shit et al., CVPR 2021. [Paper](https://arxiv.org/abs/2003.07311) | Skeleton/mask overlap and a differentiable soft-skeleton loss for connected structures. | Supervised segmentation masks; authors report improved connectivity across vessel, road, and neuron datasets. | Supports a practical structure-sensitive loss. It does not by itself establish that the right labeled diagram endpoints connect; diagram node outlines also differ from tubular masks. Treat it as borrowed supervision, not our theoretical contribution. [Author implementation](https://github.com/jocpae/clDice) | B for paper claims; C for implementation |
| 4 / High | **Region-Adaptive Sampling for Diffusion Transformers**, Liu et al., 2025 preprint; official repository identifies CVPR 2026. [Paper](https://arxiv.org/abs/2502.10389) | Training-free regional sampling, reusing earlier noise estimates in slow-update regions. | Adapts existing models without retraining; authors report quality/speed improvements in text-to-image generation. | Regional computation and importance selection predate this project. Consider a feasible sampler control for expanded experiments; the official large-model/CUDA path is not a direct Mac baseline. [Project](https://microsoft.github.io/RAS/), [repository](https://github.com/microsoft/RAS) | B; C for implementation |
| 5 / High | **TRaM-VSR: Importance-Aware Token Routing and Merging for One-Step Diffusion Video Super-Resolution**, Gao et al., July 2026 preprint. [Paper](https://arxiv.org/abs/2607.22231) | Routes/merges tokens using motion, semantic cues, and a planner; retains structurally important tokens in a detailed stream. | Uses a one-step video diffusion setting; abstract reports accelerated restoration with preserved quality and temporal consistency. Exact resource requirements were not extracted. | Particularly relevant prior art against a broad claim of “protect structural tokens in diffusion restoration.” Our potential distinction is explicit diagram-edge correctness with per-step MLP routing, without video/text priors. Full method comparison remains necessary before claiming novelty. | B; HTML available |
| 6 / High | **ARF: Arbitrary Routing Framework for All-in-One Image Restoration**, Xu et al., IEEE Transactions on Cybernetics 55, 5963–5974, 2025. [Fetched record](https://consensus.app/papers/arf-arbitrary-routing-framework-for-allinone-image-xu-gao/f079be962ab25c019763577180372849/) | Routes through subnetworks selected for restoration-task complexity using architecture search. | Requires backbone training and task-specific architecture search; abstract reports better PSNR and lower computation than AirNet. | Adaptive restoration itself is prior art. It differs from a connection-supervised, timestep-conditioned token policy. Publisher full text was unavailable; exact training costs and results were not independently checked. DOI discovery: 10.1109/TCYB.2025.3604401. | B, fetched record only |
| 7 / High | **Palette: Image-to-Image Diffusion Models**, Saharia et al., 2021 preprint, revised 2022. [Paper](https://arxiv.org/abs/2111.05826) | Image-conditioned diffusion for colorization, inpainting, uncropping, and JPEG restoration. | Paired training; authors' setup uses batch 1,024 and one million steps, reporting improvements over regression/GAN references on their tasks. | Use its conditioning principle, not its full-scale training recipe. Its evaluation is not a test of labeled diagram connectivity. Our compact transformer should be described as an implementation inspired by conditional diffusion, not a reproduction of Palette. [Full text](https://arxiv.org/html/2111.05826v2) | A; venue metadata not verified in this review |
| 8 / Medium–high | **PixRestore: Unified Image Restoration via Pixel Diffusion Transformer**, Sun et al., August 2026 preprint. [Paper](https://arxiv.org/abs/2608.16793) | Pixel-space flow matching for restoration, feature reliability conditioning, and one-step fine-tuning. | Trains on a large restoration corpus, uses DINO-based supervision; abstract reports a roughly 50M-parameter one-step model. | Pixel-space diffusion restoration and efficient single-step generation are already studied. It strengthens the case for simpler/one-step controls. A 50M-parameter external model with extra pretraining is not a budget-matched pilot reference. | B; HTML available |
| 9 / Medium, essential control | **Denoising Diffusion Implicit Models**, Song, Meng, and Ermon, 2020 preprint. [Paper](https://arxiv.org/abs/2010.02502) | A non-Markovian diffusion formulation enabling faster sampling using a compatible training objective. | Supports changing inference trajectories without retraining a new model for every sampling budget. | Fewer-step sampling is a mandatory alternative to routing. Our exact DDIM implementation and compatibility must be verified; this review does not claim a reproduction of its published benchmarks. | B |
| 10 / Medium, essential control | **U-Net: Convolutional Networks for Biomedical Image Segmentation**, Ronneberger, Fischer, and Brox, 2015. [Paper](https://arxiv.org/abs/1505.04597) | Encoder/decoder with skip connections for pixel prediction. | Supervised segmentation in the original work. | Adapt a small U-Net for restoration as required by the brief. The original segmentation results do not show how well this proposed restoration baseline will perform. Its purpose is to challenge the need for diffusion. | B |

### Version reconciliation

The DyDiT family also includes **DyDiT++: Diffusion Transformers with Timestep and Spatial Dynamics for Efficient Visual Generation**, Zhao et al., arXiv:2504.06803v4, revised January 2026. Its metadata reports TPAMI acceptance on 9 January 2026. It extends the family to flow matching, video/text-to-image, and parameter-efficient adaptation. Count this with DyDiT when discussing overlap, rather than as independent evidence for the same original result. The January 2026 arXiv title supersedes the older title still used in the repository. Publisher issue/DOI was not verified. [Current paper record](https://arxiv.org/abs/2504.06803)

## Separate implementation comparison

All checked 12 September 2026. These are publicly accessible research implementations, not verified production deployments. Repository existence is confirmed; current maintainership and latest commit dates were not audited. Licensing must be checked at the exact source revision before incorporating code.

| Priority / fit | Project / owner | Verified capability and relevance | Availability / license evidence | Limitation |
| --- | --- | --- | --- | --- |
| 1 / Very high | [DyDiT / Alibaba DAMO Academy](https://github.com/alibaba-damo-academy/DyDiT), plus [original NUS repository](https://github.com/NUS-HPC-AI-Lab/Dynamic-Diffusion-Transformer) | Official references for dynamic feedforward execution and the extended model family. | Public source; extended repository root did not expose a license in the inspected listing. Reuse license remains unresolved. | Useful algorithmic reference; no Mac compatibility or independent reproduction verified. |
| 2 / High | [clDice / jocpae](https://github.com/jocpae/clDice) | Author implementations of hard metric, soft skeleton, and PyTorch loss. | Public source; MIT license displayed. | A segmentation loss component, not a complete diagram restoration or labeled-edge evaluator. |
| 3 / High | [RAS / Microsoft](https://github.com/microsoft/RAS) | Regional diffusion-sampling reference with examples for existing image generators. | Public source; MIT license displayed. | Installation/examples use FlashAttention and CUDA; do not assume the published implementation runs on MPS. Any adaptation must be named and documented. |

## Critical synthesis and design implications

**Allocation is established; the objective remains the opportunity.** Dynamic region allocation is supported by several different mechanisms in the reviewed work. A claim that “important tokens deserve more compute” would overlap strongly. Our narrower inference is that explicit *direct-edge recovery at a given measured latency* remains a potentially useful objective to study in this constrained domain. The search did not establish that this exact combination is unstudied. [DyDiT](https://arxiv.org/abs/2410.03456), [DC-DiT](https://arxiv.org/abs/2603.06351), [TRaM-VSR](https://arxiv.org/abs/2607.22231)

**Restoration is not automatically a reason to use many diffusion steps.** The reviewed literature includes both conventional adaptive restoration and one-step diffusion restoration. Whether our diffusion model contributes useful diagram accuracy must therefore be measured against simple restoration and fewer-step inference. Their published results on other domains cannot answer that question for us. [ARF record](https://consensus.app/papers/arf-arbitrary-routing-framework-for-allinone-image-xu-gao/f079be962ab25c019763577180372849/), [PixRestore](https://arxiv.org/abs/2608.16793)

**Topology proxies and edge correctness answer different questions.** The clDice work motivates a structure-sensitive training signal. Our graph labels identify which named nodes should be directly linked, making edge precision/recall and exact-graph accuracy necessary outcome measures. This is a proposed evaluation distinction, not a refutation of clDice's theory. [clDice](https://arxiv.org/abs/2003.07311)

**Reported accelerator speedups do not transfer automatically.** Model size, kernel support, and routing overhead differ from this Mac pilot. Keep a small native PyTorch architecture, measure real end-to-end cost, and separate operation-count improvements from elapsed-time improvements. [DyDiT experiment setup](https://arxiv.org/html/2410.03456v2), [RAS implementation](https://github.com/microsoft/RAS)

## Defensible positioning to test

Proposed contribution: **a connection-supervised computation policy for conditional diagram restoration, evaluated using direct-edge errors and measured latency against matched generic routing, loss-only, and conventional restoration controls.**

Potential accompanying artifact: a reproducible diagram corruption generator and a validated direct-edge evaluator with counterexamples to image-similarity-only assessment. Their usefulness and distinctiveness must also be established; creating a synthetic dataset alone does not guarantee a publishable contribution.

Do not claim the first adaptive diffusion transformer, first structural diffusion restoration method, guaranteed preservation of graph topology, or practical speedup without the corresponding evidence. Before the method freezes, compare the full details of DyDiT++, TRaM-VSR, PixRestore, and any later directly relevant paper against the exact proposed supervision. The next literature pass should also look more deeply at graphics-recognition restoration datasets and direct graph extraction.

The linked paper rows and version note form the initial bibliography. Keep citation metadata tied to the accessed version; produce venue-specific BibTeX after the final related-work selection.

## Literature refresh for the benchmark framing (13-14 September 2026)

Everything above was scoped for the routing method, dropped after Gate G2 (see
[docs/01_RESEARCH_SCOPE.md](01_RESEARCH_SCOPE.md)'s amendment). This section repositions
against diagram-benchmark and graph-aware-evaluation papers rather than efficient-diffusion
papers, per [docs/03_PROJECT_PLAN.md](03_PROJECT_PLAN.md)'s revised Stage 12. A first pass
(13 September, web search only) is now followed by a full-text pass on the closest match
(14 September). Evidence labels follow the original review's convention: **A** = full-text
inspected, **B** = abstract/summary or search-tool-synthesized only.

| Paper | Task | Relation to this project | Evidence |
| --- | --- | --- | --- |
| **DiagramNet**, arXiv:2605.01338 (2026) | Recognition/QA over system-level chip diagrams; 10,977 connection annotations across Listing/Localization/Connection/Circuit-QA tasks. | Closest dataset match on "diagrams with connection annotations," but it's a recognition/parsing benchmark on *clean* diagrams, not a restoration benchmark on *damaged* ones, and reports no graph-vs-pixel evaluation contrast. Different task; useful for framing "diagram+graph benchmarks exist" but not a competitor. | B |
| **SciFlow-Bench**, arXiv:2602.09809v3 (Feb–Jun 2026, arXiv preprint only — no conference/journal acceptance found) | Evaluates text-to-image *generation* of scientific diagrams via a **hierarchical multi-agent inverse parser**: a "Cognitive Planning" stage extracts method descriptions into visual prompts; three concurrent "Fine-Grained Perception" agents (Environment Curator, Shape Hunter, Text Spotter) extract nodes via segmentation and OCR; a "Structural Reasoning" stage (Topology Coder → Mermaid IR → Graph Architect) produces the final graph, compared to a canonical ground truth. Authors: Tong Zhang, Honglin Lin, Zhou Liu, Chong Chen, Wentao Zhang. Code: `github.com/Tong-0302/SciFlow-Bench`. | **The closest prior argument to this project's Finding 04** (appearance metrics hide structural failures) — explicit: "structural recoverability rather than visual similarity alone." But the mechanism is entirely different from this project's: a multi-agent VLM/OCR pipeline with **semantic node matching via sentence embeddings** (cosine similarity on text descriptions) and **directed** dependency graphs, versus this project's single deterministic function using **geometric** node matching (known bounding boxes, no text/semantics) and **undirected** connectivity graphs. Node/edge F1 weighted 40/60 into their graph-level score (own design choice; this project reports edge-F1 alone as the primary endpoint). **Also independently found diffusion underperforming on structural fidelity**: "SDXL and PixArt-Σ exhibit consistently weak structural recoverability... pure diffusion-based generators... often fails to reliably preserve directed dependencies," with autoregressive VLMs (Gemini 3 Pro Image) as their strongest performer — a different task and a different "winner" than this project's small U-Net, but the same qualitative pattern (diffusion trailing on structural correctness), independently observed. Not scooped — different mechanism, different task (generation vs. restoration), different graph convention — but must be cited and distinguished explicitly, and this parallel finding is worth citing as corroborating evidence from an adjacent domain. | **A** |
| **ERQA** (Edge-Restoration Quality Assessment), arXiv:2110.09992 | Edge-sensitive quality metric for video super-resolution. | An appearance metric already tuned to be more edge-sensitive than Dice/PSNR/SSIM. Worth adding as a comparison point in the Stage 7 divergence analysis — it may narrow, but is unlikely to close, the gap Finding 04 found, since it is still a *pixel/gradient*-level edge metric, not a graph-topology one. Not yet tested. | B |
| **DocRevive** (introducing **OPRB**, the Occluded Pages Restoration Benchmark), Purkayastha et al., **CVPR 2026 Workshop (MULA — Multimodal Understanding for Long-form Analysis)**, arXiv:2604.10077. 30,078 degraded document images (23,212 text-only, 6,866 with figures/diagrams), 6 occlusion classes (Black Ink, Burnt, Whitener, Dust, Scribble, Stamp). | Document *occlusion* restoration (opaque/semi-transparent overlays removed), with word-level supervision — a different corruption model from this project's noise/blur/gap, and no graph/connectivity evaluation; diagrams are a ~23% minority of the benchmark, not its focus. Venue/authors now verified (was previously an open item). Adjacent evidence that document-restoration benchmarks are an active, publishable category, not a direct competitor. | B (venue/authors verified via search; full-text methods not yet read) |
| Enginuity, arXiv:2601.13299 / 2606.03410 | Vision-language understanding benchmark for engineering diagrams. | Recognition/VQA, not restoration; same category as DiagramNet — evidence that diagram+graph benchmarks are an active area, not a direct competitor. | B |
| **Vision-Based Topology-Consistent Structural Parsing of Hand-Drawn Circuit Diagrams**, *Sensors* 26(11):3440, 2026 (peer-reviewed journal; PMC13259090). Benchmark: 1,317 hand-drawn circuit diagrams (972 photographed pre-existing + 345 newly drawn), with natural hand-drawing and camera-acquisition artifacts retained. | Parses hand-drawn circuit photographs into a netlist via a staged pipeline (detection, OCR, node/terminal prediction, wire enhancement, **connectivity reasoning**, endpoint semantics, netlist generation), reporting 95.14% strict image-level success. | Closest found example of a **real, non-synthetic, connectivity-focused diagram benchmark** — but it's parsing/recognition (photo → netlist), not restoration of a damaged diagram back to a clean one, and it doesn't compare architecture families (diffusion vs. CNN) or use appearance-vs-graph divergence as an argument. Relevant precedent for "real hand-drawn diagram acquisition is a tractable, publishable benchmark category" — worth citing as motivation for this project's still-open renderer/acquisition-shift item, not as a direct competitor. | B |

**Counter-to-conventional-wisdom framing worth using in the paper**: broader diffusion image-generation literature generally argues diffusion models produce *better* structural fidelity than CNNs (avoiding CNN's tendency to average over plausible outputs). This project's finding — a small CNN beats a conditional diffusion transformer on structural/graph fidelity specifically — runs counter to that general intuition, which is a positioning strength, not a weakness: it says the "diffusion is more structurally faithful" heuristic doesn't transfer to this constrained, exact-connectivity restoration setting, and SciFlow-Bench's independent structural-recoverability finding for diffusion-based *generation* offers a second, adjacent data point for the same qualitative pattern.

### What this changes about the novelty statement

The original review's claim — a reproducible diagram corruption generator and a validated
direct-edge evaluator "with counterexamples to image-similarity-only assessment" — is
**not scooped**, but SciFlow-Bench shows the core motivating argument (visual similarity can
hide structural failure) is being made independently and concurrently in the adjacent
diagram-generation literature, using a categorically heavier mechanism (multi-agent VLM
parsing vs. this project's single deterministic geometry function). The defensible, narrower
claim: this project demonstrates that argument **empirically, with a lightweight
geometry-assisted extractor requiring no learned parser or VLM**, for the *restoration*
setting specifically, and pairs it with a rigorous multi-seed, confound-controlled,
multi-scale, distribution-shift-tested finding that a small CNN beats diffusion for this
structural task — a comparison SciFlow-Bench does not make (it evaluates generation quality
across many models, not a controlled restoration-architecture ablation).

### What still needs doing before drafting related work

1. ~~Full-text read of SciFlow-Bench's evaluation section~~ — **done above.**
2. ~~Verify OPRB's exact venue, authors~~ — **done above** (Purkayastha et al., CVPR 2026
   Workshop MULA, arXiv:2604.10077). OPRB's own restoration-evaluation methodology (beyond
   venue/authors) has not been read in full text.
3. Search specifically for diffusion-vs-CNN restoration comparisons on tasks with an
   explicit graph/topology ground truth (vessel/road segmentation with clDice is the closest
   found so far, already in the original review) — **still not found** in the diagram
   domain specifically after two search passes; this is either a genuine gap this paper
   fills or a sign the queries need broadening (e.g., circuit-diagram or wiring-diagram
   restoration specifically, or citation-chasing SciFlow-Bench's Table 4 comparison).
4. Read SciFlow-Bench's Table 4 in full (diffusion vs. autoregressive-VLM vs. code-driven
   comparison) — currently known only via search-tool synthesis, not directly inspected.
5. A citation-graph check (who cites SciFlow-Bench, DiagramNet; what they cite) has not been
   done — search-engine discovery only, not systematic.
6. Repeat this search closer to the submission date given how recent every hit above is
   (all 2026, several within the last two months) — this literature is moving fast, and
   SciFlow-Bench specifically was still receiving new versions (v3, June 2026) after this
   review.

Search log: see [docs/07_SEARCH_LOG.md](07_SEARCH_LOG.md)'s 13 September addendum for exact
queries.
