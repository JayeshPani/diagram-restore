# Research scope

Version 0.1 · 12 September 2026 · proposed settings, not experimental findings

**Working title:** Connectivity-Aware Adaptive Diffusion for Diagram Restoration.

**Research question:** Can connectivity-aware routing reduce the computation of a diffusion model while preserving connections in restored diagrams?

**Problem to establish.** A restored diagram can look similar to its clean reference while losing a short connection or inventing a shortcut. We will test whether these mistakes occur in our controlled task, and whether reducing denoising computation makes them worse. Their prevalence in this dataset is currently unknown.

**Input and output.** The model receives only a damaged grayscale raster image. The target is a clean black-and-white image with the original direct connections. The initial domain contains boxes, circles, and undirected connecting lines at 64 × 64 pixels, with mild noise, blur, and short line gaps. Graph annotations support training and evaluation; node locations and true edges are never additional inference inputs. The pilot uses separated connector paths with no ambiguous crossings or free-standing junctions. Later junction experiments require an explicit graph convention.

**Proposed improvement.** Start with a compact conditional diffusion transformer in pixel space. Preserve shared attention and select the tokens that receive feedforward updates. Train a router using image features and timestep, with supervision that emphasizes connectors, line gaps, and node attachment regions. Skipped tokens follow the residual path. Benchmark actual sparse execution, including routing overhead.

The bypass mechanism itself is prior work: DyDiT already preserves attention while routing tokens around MLP blocks. Our proposed contribution is the connection-specific allocation criterion and its controlled evaluation in diagram restoration. This difference is a hypothesis to test, not an established novelty claim. [DyDiT, §3.3](https://arxiv.org/html/2410.03456v2)

**Essential comparisons.** Damaged input; simple morphological repair; small U-Net, including a structural-loss version; a deterministic transformer using the diffusion backbone; dense conditional diffusion; fewer diffusion steps; generic routing; generic routing with the same structural restoration loss; a dense model with that loss; foreground/edge-based routing; the complete proposal and component removals. An oracle connector policy is a diagnostic, not a deployable competitor.

**Proposed paper success criteria.** Freeze these before final test evaluation. A candidate must satisfy either: (A) at least 15% lower median end-to-end latency than the best eligible dense diffusion configuration, with direct-edge F1 no more than 1 percentage point worse and exact-graph accuracy no more than 2 points worse; or (B) at least 2 points better direct-edge F1 within a ±5% measured runtime band. Also demonstrate an advantage over generic routing with the same structural loss. Report edge precision/recall, invented and missing edges, variability across at least three training seeds, and generalization. Confidence intervals must support the claim; a point estimate alone is insufficient. These are project decision thresholds, not universal publication standards.

**Practical relevance.** U-Net remains an essential reference. If it matches or exceeds diffusion's connectivity accuracy at lower latency throughout the useful operating range, report that outcome and reconsider the diffusion contribution. If only operation counts improve, claim computational savings rather than acceleration on the Mac.

**Exclusions.** No OCR, arrow-direction prediction, complex circuit symbols, text generation, full-page document understanding, user interface, or large pretrained image generator in the pilot. This project belongs to document image analysis/graphics recognition. Any course requirement for a text-mining component needs a separate scope decision.

**First checkable result.** A reproducible 500/100/100 clean-diagram split, inspected clean/damaged examples, and a tested evaluator that distinguishes a direct A–C edge from the indirect path A–B–C. No router work precedes a credible baseline and oracle result.
