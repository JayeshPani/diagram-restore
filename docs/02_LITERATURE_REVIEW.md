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

## Literature refresh for the benchmark framing (13 September 2026)

Everything above was scoped for the routing method, dropped after Gate G2 (see
[docs/01_RESEARCH_SCOPE.md](01_RESEARCH_SCOPE.md)'s amendment). This section is a first pass
at the refresh [docs/03_PROJECT_PLAN.md](03_PROJECT_PLAN.md)'s revised Stage 12 calls for —
positioning against diagram-benchmark and graph-aware-evaluation papers rather than
efficient-diffusion papers. Web search only, no full-text inspection; treat as **evidence
label B** (abstract/summary level) throughout, weaker than the original review's A-labeled
sections. A full pass (citation graph, full-text methods) is still required before drafting
related work.

| Paper | Task | Relation to this project |
| --- | --- | --- |
| **DiagramNet**, arXiv:2605.01338 (2026) | Recognition/QA over system-level chip diagrams; 10,977 connection annotations across Listing/Localization/Connection/Circuit-QA tasks. | Closest dataset match on "diagrams with connection annotations," but it's a recognition/parsing benchmark on *clean* diagrams, not a restoration benchmark on *damaged* ones, and reports no graph-vs-pixel evaluation contrast. Different task; useful for framing "diagram+graph benchmarks exist" but not a competitor. |
| **SciFlow-Bench**, arXiv:2602.09809 (2026) | Evaluates text-to-image *generation* of scientific diagrams by inverse-parsing the generated image back into a structured graph and comparing it to a canonical ground-truth graph — explicitly "structural recoverability rather than visual similarity alone." | **The closest prior argument to this project's Finding 04** (appearance metrics hide structural failures) — but for generation, not restoration, using directed dependency graphs with semantic node matching and a *learned inverse-parser*, versus this project's undirected connectivity graphs with geometric node matching (known bounding boxes) and a raster-tracing extractor. Different mechanism, same motivating claim; must be cited and distinguished explicitly, not treated as scooped. Node/edge-level precision-recall-F1 with edge weighted 60% of the graph score (their design choice; this project currently reports edge-F1 as the sole primary endpoint, node-level not separately scored) and a "path-aware semantic matching" edge-correctness rule (their directed-dependency setting; not applicable to this project's undirected, geometrically-anchored edges). |
| **ERQA** (Edge-Restoration Quality Assessment), arXiv:2110.09992 | Edge-sensitive quality metric for video super-resolution. | An appearance metric already tuned to be more edge-sensitive than Dice/PSNR/SSIM. Worth adding as a comparison point in the Stage 7 divergence analysis — it may narrow, but is unlikely to close, the gap Finding 04 found, since it is still a *pixel/gradient*-level edge metric, not a graph-topology one. Not yet tested. |
| Occluded Pages Restoration Benchmark (OPRB) (found via search, exact venue/authors not yet verified) | Document restoration benchmark, 30,078 degraded document images, ~23% containing figures/diagrams. | Adjacent (document restoration, not diagram-specific graph connectivity), and evaluation methodology not yet inspected. Flag for full-text check before citing. |
| Enginuity, arXiv:2601.13299 / 2606.03410 | Vision-language understanding benchmark for engineering diagrams. | Recognition/VQA, not restoration; same category as DiagramNet — evidence that diagram+graph benchmarks are an active area, not a direct competitor. |

### What this changes about the novelty statement

The original review's claim — a reproducible diagram corruption generator and a validated
direct-edge evaluator "with counterexamples to image-similarity-only assessment" — is
**not scooped**, but SciFlow-Bench shows the core motivating argument (visual similarity can
hide structural failure) is being made independently and roughly concurrently in the
adjacent diagram-generation literature. The defensible, narrower claim: this project
demonstrates that argument **empirically, with a lightweight geometry-assisted extractor
(no learned parser required)**, for the *restoration* setting specifically, and pairs it
with a rigorous multi-seed, confound-controlled finding that a small CNN beats diffusion for
this structural task — a comparison SciFlow-Bench does not make (it evaluates generation
quality, not restoration architecture choice).

### What still needs doing before drafting related work

1. Full-text read of SciFlow-Bench's evaluation section — confirm exactly how its inverse
   parser works and whether its "structural recoverability" argument is empirically
   demonstrated (a divergence table like this project's) or only motivated.
2. Verify OPRB's exact venue, authors, and evaluation metrics — currently known only from a
   search snippet.
3. Search specifically for diffusion-vs-CNN restoration comparisons on tasks with an
   explicit graph/topology ground truth (vessel/road segmentation with clDice is the closest
   found so far, already in the original review) — the diagram-specific version of this
   comparison was not found in this pass, which is either a genuine gap this paper fills or
   a sign the queries need broadening.
4. Repeat this search closer to the submission date given how recent every hit above is
   (all 2026, several within the last two months) — this literature is moving fast.

Search log: see [docs/07_SEARCH_LOG.md](07_SEARCH_LOG.md)'s 13 September addendum for exact
queries.
