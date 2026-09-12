# Experiment protocol

Version 0.1 · 12 September 2026. All numerical settings below are **proposed development defaults**. Record amendments and freeze final choices using validation data before evaluating the reserved test set. No experimental results exist yet.

## 1. Questions and order of evidence

H1: connection errors occur in diagram restoration, and reducing diffusion computation worsens them in at least one useful operating range.

H2: at the same executed feedforward budget, protecting true connector regions gives better direct-edge recovery than random selection; protecting gaps/attachments also needs to be compared with generic foreground protection.

H3: a router trained to identify connection-relevant regions from inference-available features improves the connectivity/runtime frontier over a generic router with the same structural restoration loss.

H4: any demonstrated advantage persists across training seeds and specified distribution shifts, with routing overhead included.

Each hypothesis can fail. Establish H1 and H2 before building the complete method. For every model, plot the U-Net reference as well as diffusion comparisons.

## 2. Pilot data specification

| Item | Initial specification |
| --- | --- |
| Clean parents | 500 train, 100 validation, 100 reserved test |
| Images | 64 × 64, one channel; black ink on white background; intensities stored explicitly |
| Nodes | 2–5 nonoverlapping boxes/circles, approximately 8–12 pixels across, with stable node IDs |
| Graph | Simple undirected graph; no self-loops or parallel edges; include absent node pairs and disconnected examples |
| Connectors | Straight or short orthogonal polylines, 1–2 pixels wide; no intersections, shared interior segments, or touches to unrelated shapes in the pilot |
| Junctions | Shape nodes may have several incident edges. Free-standing branching/crossing points are outside pilot v1. |
| Gap corruption | Typically 1–3 pixels along a line; retain visible evidence on both sides; never erase a whole edge; cap removed length at 20% of that edge |
| Noise | Gaussian intensity noise with standard deviation initially 0–0.03 on a [0,1] image |
| Blur | Gaussian blur standard deviation initially 0–0.6 pixels |
| Initial strata | 10% clean controls, 30% noise-only, 30% blur-only, 30% short-gap with optional mild noise/blur; balance by split |
| Dataset randomness | Separate graph, renderer, corruption, and split seeds; persist every seed |

These dimensions are starting points. Reject geometry that cannot be rendered unambiguously with the chosen line width/clearance. Use at least a three-pixel clearance from nonincident geometry where possible, and validate the actual raster rather than trusting continuous coordinates. Avoid correlations such as “circles always connect” or “the nearest nodes always connect.”

First generate exactly one fixed damaged version per parent for the pilot. Additional training corruptions may be sampled later from training parents only. Validation/test variants must be fixed in manifests and aggregated by parent. Do not count multiple corruptions of one clean diagram as independent diagrams.

### Annotation contract

Every record contains `diagram_id`, `parent_family_id`, split, image paths, image/geometry hashes, node IDs/types/centers/bounding geometry, true undirected edge pairs, connector polylines and masks, attachment regions, clean skeleton, gap mask, corruption parameters/seeds, and renderer version. Gap masks and route supervision are privileged training/diagnostic labels.

The deployment API is conceptually `restore(damaged_image, budget, seed) -> restored_image`. Its input record must omit node locations, clean images, graph edges, connector masks, and corruption locations. The ordinary training loader separates image conditioning from supervision. Add a check that inference succeeds when all annotation fields are absent.

### Leakage prevention and difficulty

Sample and assign **clean parents** to splits before generating damage. Keep re-renders, crops, rotations, and other derived variants of a parent in the same split. Check exact clean-image/geometry duplicates across splits; group deliberately reused layout templates where they encode effectively the same geometry. Distinct graphs/layouts can legitimately share abstract topology in the in-distribution split; topology-family holdout is a separate, explicitly harder test.

Do manual development inspection on training/validation examples. Keep final test predictions and labels out of tuning and failure-driven redesign. Clean-test generator assertions may check annotation consistency without being used to select model settings. If the pilot test is later opened for exploration, retire it as a development set and generate a new reserved publication test set before further development.

Short gaps do not guarantee identifiability. Include nearby unconnected lines/ports so “connect every nearby endpoint” can fail, but avoid systematically contradictory annotations for indistinguishable observations. Audit ambiguous cases using a written rule independent of model predictions. Report their incidence; use a separately labeled ambiguity stress set if necessary.

## 3. Build the evaluator before training

The primary task is recovering the **direct edge set**, not merely connected components or reachability. In the graph A–B–C, A and C are reachable but there is no direct A–C edge. A whole-image connected-component test would miss this distinction.

### Controlled extraction algorithm

1. Convert the restored grayscale image to an ink mask using a threshold fixed from validation calibration. Start development at 0.5; evaluate sensitivity to 0.4 and 0.6 later.
2. Use known node geometry to remove node interiors/outlines from the connector-search image. Define a narrow exterior contact band for each node. Retain connector contacts at those bands. This is an explicit **reference-node-assisted evaluator**, not end-to-end node detection.
3. Skeletonize or trace the remaining ink with a fixed pixel-neighborhood convention. Start with eight-neighbor connectivity and test diagonal near-touches explicitly. Do not apply gap-closing as an invisible evaluator preprocessing step.
4. Trace connector components and identify which node contact bands they touch. A component touching two distinct nodes predicts that unordered pair. Components touching zero or one node are dangling/spurious fragments, recorded separately.
5. If a predicted merged component touches more than two nodes in the no-junction pilot, emit every pair connected through that component outside the node masks and flag the component as an invalid merge. These extra predicted pairs count as false edges where absent from the reference. Do not quietly discard difficult components.
6. Finish prediction extraction **before** reading the true edge list. Compare the extracted set against the reference afterward.

The extractor may use reference node location/shape support consistently for every method. It must not use true edge pairs, true polylines, connector masks, gap masks, or true endpoint assignments to decide what is present. In particular, checking only ground-truth connector corridors cannot reveal invented edges and is unsuitable as the primary evaluator.

The assisted protocol is a limitation: a severely destroyed or moved node may be hidden by reference geometry. Report node-boundary/image quality and audit such cases. A later independent node detector is an extension, not a prerequisite secretly assumed in pilot claims.

### Mandatory fixtures and checks

Use at least 40 deterministic fixtures: intact line; one-pixel gap; removed edge; added absent edge; wrong attachment; nearby unconnected lines; diagonal near-touch; two separate incident edges at one shape; A–B–C without A–C; cycle with one edge cut; disconnected graph; blank prediction; all-ink prediction; spurious isolated fragment; merged paths; small boundary offset; and zero-edge graph. Specify each exact expected edge set manually.

All fixtures must pass. Clean generated images must recover their recorded edge sets exactly within supported geometry; repair the renderer/extractor contract if they do not. Inspect at least 30 development predictions per major model family, including random examples and every extraction warning. Save overlays and a separate extraction-error count. Threshold sensitivity and independent manual spot checks must accompany final metrics.

### Metrics

For true edges E and extracted edges P: TP = |E ∩ P|, FP = |P − E|, FN = |E − P|.

Report micro edge precision, recall, and F1 by summing TP/FP/FN across images; missing edges per image; invented edges per image; and exact-graph accuracy, the fraction with P = E. Also report per-image F1 distribution and metrics per corruption stratum, rather than only a pooled average.

For per-image zero denominators: if both edge sets are empty, F1 = 1; if exactly one is empty, F1 = 0. Precision/recall with a zero denominator are marked undefined in detailed records, while their aggregate micro values are computed from summed counts. Do not allow these conventions to hide blank or all-ink failures.

Appearance metrics: foreground Dice, boundary F1 with a fixed one-pixel tolerance at 64 × 64, and PSNR/SSIM on grayscale output. Fix range and implementation parameters. clDice is a secondary structure metric. Background-dominated image scores do not replace edge recovery. Graph extraction is evaluation overhead and is timed separately from restoration inference.

## 4. Compact models and training recipe

| Component | Proposed default |
| --- | --- |
| U-Net | Three resolution levels, base width 32, grayscale input/output; small enough for local training |
| Conditional DiT | Pixel space, 4 × 4 patches = 256 tokens at 64 × 64; width 128; 6 blocks; 4 attention heads; MLP ratio 4 |
| Conditioning | Patchify the concatenation of current noisy target and damaged observation; timestep embedding; fixed positional information |
| Model size | Aim for a few million parameters; count the implemented parameters before training |
| Training diffusion | Epsilon prediction; 1,000-step cosine noise schedule; one randomly sampled timestep per training example |
| Sampling | Compatible DDIM implementation; initially 10, 20, and 50 denoiser evaluations, eta = 0; same initial noise across paired comparisons |
| Optimizer | AdamW; initial learning rate 1e-4; batch 8, reduced to 4 if benchmarking requires it; gradient clipping 1.0 |
| Precision | Float32 initially; any later lower-precision path needs numerical and operator checks |
| Training seeds | Development: 7; confirmatory: 7, 17, 27 |

The conditional-diffusion principle follows Palette; using fewer compatible sampling steps follows DDIM. This recipe is a small local design, not a claim to reproduce either published setup. [Palette](https://arxiv.org/abs/2111.05826), [DDIM](https://arxiv.org/abs/2010.02502)

With clean target x0 and damaged observation y, sample epsilon ~ N(0,I) and form xt = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) epsilon. Train epsilon_theta(xt,y,t) to predict epsilon. The restored image is obtained by reverse sampling from noise conditioned on y; the training diffusion noise is distinct from the observed diagram corruption.

Use normalized images consistently. Base diffusion loss is noise MSE plus a modest reconstruction term on the implied x0 estimate. Initially apply the reconstruction/structural terms only where alpha_bar_t ≥ 0.1 to avoid amplifying high-noise prediction errors through division by sqrt(alpha_bar_t). Report this choice and use it identically in matched controls. U-Net uses a direct reconstruction loss, with an optional equivalent structural term.

Overfit 8–16 training diagrams first with fixed corruption and fixed evaluation noise. Check falling loss and visibly correct restored edges; aim for direct-edge F1 ≥ 0.95 on this diagnostic set. Failure is a reason to debug normalization, scheduler, conditioning, rasterization, or capacity before full pilot training. Tiny-set performance is never reported as generalization.

A deterministic transformer control uses the same backbone, damaged observation, a zero second channel, fixed time embedding, and a direct restoration objective. It tests whether transformer capacity, rather than iterative denoising, explains any U-Net gap.

## 5. Oracle diagnostic before a learned router

Start with the trained dense backbone. First inspect several equal-token masks without retraining as a sensitivity probe. Because unfamiliar masking can itself degrade a dense model, confirm the finding on a **single shared mask-tolerant checkpoint** fine-tuned with random/varied keep masks. All diagnostic policies use the same weights, examples, timesteps, sampling noise, and executed keep counts.

At keep ratios 0.50 and 0.75, compare random masks (several random draws), regular spatial masks, observation-edge ranking, all-foreground oracle ranking, all-connector oracle ranking, and connector-gap/attachment oracle ranking. If there are more priority patches than the budget, subsample deterministically under the same rule; if fewer, fill from remaining patches with the same fallback rule.

Ground-truth connector/gap locations are allowed **only** in oracle diagnostics or training targets. These are privileged-location heuristics, not guaranteed optimal allocation oracles. Their comparison establishes whether this location information helps under the chosen intervention; it cannot prove a learnable router will match them. Report operation budgets, not deployable oracle speedup.

Proceed when a consistent validation advantage is visible across multiple diagram types and masks; a provisional useful effect is at least 2 F1 points over random at one nontrivial budget. The comparison with all-foreground/observation-edge ranking is necessary to decide whether connection-specific information adds value.

## 6. Smallest proposed router

Keep attention dense. After attention and before selected MLP sublayers, compute a score for each token from its current feature and timestep embedding. Initially route only the last four of six blocks; retain the first two fully. Use one small linear/MLP head per routed block. The conditioning image remains available through the backbone features.

At a chosen keep fraction r, select k = ceil(rN) tokens using top-k scores, gather them into a compact tensor, execute the MLP on that tensor, and scatter the updates back. Every other token follows the residual bypass. Test r = 0.50, 0.75, and 1.00 first. This gives content/timestep-dependent *selection* at a controlled budget; it does **not** claim learned variation in the number of selected tokens. Variable per-step budgets are a later extension if justified.

### Concrete supervision to investigate

Build a soft patch-priority target from training annotations: background = 0; intact connector region = 0.25; connector-to-node attachment region = 0.6; short missing-connector gap with a one-pixel margin = 1.0. Resolve overlaps by maximum and max-pool into patch targets. These are starting weights. They favor continuity/attachments over box outlines, but remain a structural proxy rather than a proof of causal edge importance.

The reconstruction/structure target is the clean image. The router target is derived offline from clean connectors and the known training corruption. Inference predicts scores from image features and time only. An all-ink target of identical capacity/supervision strength is a necessary control against the explanation that any foreground supervision would work.

Proposed objective:

```text
L = L_noise + lambda_rec * L_reconstruction
    + lambda_sk * (1 - soft_clDice)
    + lambda_route * BCE(router_probabilities, patch_priority)
    + lambda_cost * (mean(router_probabilities) - r)^2
```

Use a validation-limited search for the lambda values, with equal trial budgets across learned variants. Compute soft-clDice on foreground probability and clean foreground; audit whether node outlines dominate it. Source and credit any reused soft-skeleton implementation. [clDice implementation](https://github.com/jocpae/clDice)

Fixed top-k already constrains executed inference cost. The probability regularizer supports gate training/calibration; it is not evidence of additional savings or a learned variable compute budget. Warm-start from a common dense checkpoint. Train across sampled keep ratios, with a short dense warm-up to avoid an abrupt masking distribution shift.

For discrete gates, use a documented straight-through soft-to-hard training estimator. A dense masked MLP path may be used during training so gradients reach gate scores; never use it to claim inference speed. Verify nonzero finite gate gradients, hard-mask forward behavior, and numerical equivalence to the gathered sparse inference path at fixed masks. Pure indexing without a surrogate gate gradient is not a valid learned-routing implementation.

## 7. Comparison matrix and fair attribution

All variants receive the same image-only inference information and use the same evaluation pipeline. “Structural loss” below means the same reconstruction-side soft-clDice term; “priority target” means separate router supervision.

| ID | Model/policy | Structural loss | Router priority target | What it tests |
| --- | --- | --- | --- | --- |
| B0 | Damaged input unchanged | None | None | Corruption severity and evaluator behavior |
| B1 | Small morphological closing/denoising baseline | None | None | Whether simple repair already solves the task; tune kernel on validation |
| U0 | Small U-Net | No | None | Required conventional restoration reference |
| U1 | Same U-Net | Yes | None | Whether cheap structural supervision is sufficient |
| T0 | Deterministic restoration transformer | Yes | None | One-pass, similar-backbone alternative to diffusion |
| D0 | Full conditional DiT, 10/20/50 steps | No | None | Dense quality and fewer-step acceleration |
| D1 | Full conditional DiT, 10/20/50 steps | Yes | None | Benefit of structural loss without routing |
| G0 | Learned generic MLP routing | No | None | Benefit of ordinary adaptive computation |
| G1 | Learned generic MLP routing | Yes | None | Closest matched control for the proposed routing supervision |
| E0 | Observation-edge-ranked fixed routing | Yes | None | Cheap importance heuristic available at inference |
| F0 | Learned MLP routing | Yes | All-ink/foreground target | Whether generic foreground supervision explains the result |
| C0 | Complete proposed routing | Yes | Connector/gap/attachment target | Full proposal |
| C1 | Proposed routing without structural restoration loss | No | Connector/gap/attachment target | Whether the skeleton loss is necessary |

“Our model without routing supervision” is **G1**, and “without either structural signal” is **G0**. Reuse those rows instead of training duplicate variants under different names. Oracle rows belong in a separate diagnostic table and never on the deployable frontier.

Use a staged comparison budget: all inexpensive baselines first; one-seed screening after G3; then at least three training seeds for U0/U1, D0/D1, G0/G1, F0, C0, and C1 if their ablations support paper claims. T0/E0 results used for a superiority claim also need repeated training where applicable. If resources do not permit this, narrow the claimed comparisons rather than presenting one lucky run as robust.

Warm-start router variants from the same seed-specific dense checkpoint. Give dense controls equal additional optimization opportunity. Report inherited pretraining, additional steps, examples seen, total training wall time, and hyperparameter-search cost separately. Keep size/augmentations/optimizer budgets comparable; equal optimizer steps alone do not imply equal computation. Give U-Net enough development effort to avoid an undertrained straw-man baseline.

Budget matching for attribution uses actual active MLP counts; budget matching for practical claims uses measured end-to-end latency. Compare fewer-step dense models on the latency frontier, not only a convenient 50-step reference. Report MACs/FLOPs with an explicit multiply-add convention; do not equate an MLP keep fraction with whole-model savings.

## 8. Timing, memory, and real operation skipping

Measure batch-one latency as the primary deployment scenario and a fixed small-batch throughput separately. Use eval/inference mode; identical resolution, precision, device placement, and sampler; fixed power settings; and an idle plugged-in machine. Randomize the order of method timing to reduce thermal/order bias.

Warm up each configuration with at least 20 complete restores. Then time 100 images in at least five repeated sessions, synchronizing MPS immediately before and after timed work. Report median, interquartile range, p95, and uncertainty for latency ratios. Include conditioning, initial noise creation, all denoising steps, router/top-k/gather/scatter, and conversion to the returned image. Report a separate device-only denoising time if useful. Exclude file decoding and graph extraction from the primary restore time, and report them separately. [MPS synchronization](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html)

Instrument MLP input sizes and dense attention costs. Demonstrate that inactive tokens are absent from MLP matrix multiplies. All-keep sparse execution must match the dense output within a validated float32 tolerance; fixed sparse masks must match the dense masked reference. Include any CPU transfers and fallback operations in the measured path.

Log process RSS plus MPS current allocated and driver allocated memory at supported checkpoints. These are different quantities and not additive dedicated-GPU memory. A sampled maximum is a **sampled high-water mark**, not an exact device peak. If precise peak instrumentation is unavailable, say so. Include model, optimizer, and activation costs in training budget reports.

Amdahl-style feasibility estimate: if fraction f of total inference time is the MLP work actually eligible for routing, keep fraction r gives an optimistic time ratio (1-f) + rf, before overhead. For example, f = 0.40 and r = 0.50 yields 0.80 before routing overhead; these numbers are illustrative, not measured. Profile f rather than assuming 50% token removal means 50% faster inference.

## 9. Generalization and confirmatory analysis

After G3/G4, target an expanded 5,000/500/1,000 clean-parent split at 64 × 64, with pilot development parents confined to training/development and a fresh reserved test set. This is a resource-dependent starting size, not a publication minimum or a statistical power guarantee. Decide final test size using development variance and the predeclared effect size before opening test predictions.

Add separate held-out sets for unseen layout families, thinner lines, stronger yet interpretable corruptions, and crowded near-misses. For 128 × 128, profile a matched model first and keep the resolution/capacity change explicit. Add higher-degree nodes and marked free-standing junctions only with an extended tested evaluator: junction dots become graph vertices, and edges terminate at shapes/dots. Never silently treat crossings as junctions or compare different graph conventions in one pooled metric.

Target 50–100 independently authored diagrams rendered by a second tool or drawn/scanned by collaborators, with paired references and manually checked direct edges. Record origin, consent/license, acquisition, registration, and exclusions. A second synthetic renderer tests renderer shift; independently scanned diagrams test an additional acquisition shift. Do not label synthetic re-rendering as real-world validation. Existing diagram datasets require inspection of both license and edge/pair annotations before use.

Freeze model settings, keep ratios, sampling counts, evaluator thresholds, seeds, and primary endpoints on validation. Select latency-matched competitors from validation timing, then evaluate the locked choices on test. No best-of-many diffusion samples: primary output uses a fixed noise seed per image; secondary stochastic analysis averages over predetermined noise seeds without oracle selection.

Report mean and standard deviation across independently trained seeds. Use paired bootstrap intervals over **clean parents**, retaining all corruptions per parent; show each seed's paired effect. For uncertainty spanning optimization variability, also use a hierarchical seed/parent analysis and acknowledge that three seeds give limited seed-level precision. Timing repetitions are not independent model-training replicates. Inspect subgroup harm and extraction failures; do not select only favorable conditions after seeing test results.

## 10. Proposed decision criteria

Primary endpoints: micro direct-edge F1 and median end-to-end restoration latency. Guardrails: edge precision/recall, invented edges, exact-graph accuracy, and extraction-error audit. Do not call a statistically nonsignificant quality difference “equivalent.”

Criterion A: at least 15% lower latency, with F1 degradation at most 1 percentage point and exact-graph degradation at most 2 points, against the validation-selected best eligible dense model. Eligibility means within those quality margins of the strongest validation dense configuration. Freeze that comparator before test.

Criterion B: at least 2 percentage points higher F1 within a ±5% runtime band against the locked latency-matched dense configuration. Also require improvement over G1/F0 at matched cost to support connection-specific routing, with the same structural loss and comparable supervision opportunity.

Use paired confidence intervals: for A, the F1-difference lower bound must exceed −0.01 and the exact-graph lower bound exceed −0.02. A claim of *at least* 15% latency reduction needs the latency-ratio upper bound ≤0.85; otherwise report the measured estimate and weaker supported bound. For B, the F1-difference lower bound must reach +0.02 and the runtime match must be supported by repeated timing. If evidence is too imprecise, report uncertainty rather than declaring success.

These proposed thresholds are a research-management choice. They may be revised with a written development-only justification before final protocol freeze, never retrofitted to final test outcomes. Meeting them does not guarantee acceptance; failing them still produces useful diagnostic evidence but may require a different paper claim.
