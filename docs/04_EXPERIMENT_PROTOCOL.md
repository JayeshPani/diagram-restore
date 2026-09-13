# Experiment protocol

Version 0.2 · 13 September 2026. Sections 1–4, 8, and 9 are revised or actively used; sections 5–7
covered the routing method dropped after Gate G2 (see
[docs/01_RESEARCH_SCOPE.md](01_RESEARCH_SCOPE.md)'s amendment and
[results/milestone2/G2_REPORT.md](../results/milestone2/G2_REPORT.md)) and are kept below,
clearly marked, as the historical record of that design rather than deleted. Sections 2–4's data
spec, evaluator spec, and model/training recipe are unaffected by the pivot and remain the
project's live defaults.

## 1. Questions and order of evidence (revised)

H1 was tested and answered before the pivot: connection errors do occur, and both are true —
reducing diffusion sampling steps did **not** measurably worsen them in the 10–50 step range at
this pilot scale, but dense diffusion trailed the small U-Net at every step count regardless (see
Gate G2). H2–H4 concerned the dropped routing method and were not pursued (no oracle or router was
built).

The revised questions for the benchmark contribution:

RQ1 *(answered, Gate G2)*: which restoration architectures dominate the direct-edge accuracy/
latency frontier on this pilot task, and is the answer robust to training budget, standard
diffusion training tricks, model capacity, and training seed? — U0/U1 dominate D0/D1 on both axes,
confirmed across 3 seeds and 4 independent debugging interventions.

RQ2 *(answered, confirmed at 3 seeds, Stage 7)*: does an appearance-similarity metric
(Dice/PSNR/SSIM) rank restoration architectures the same way the direct-edge evaluator does, or
does it hide the gap RQ1 found? **It hides it, consistently.** Mean Dice's range across all seven
compared methods is 4.48× ± 0.60 narrower than edge-F1's range, across seeds 7/17/27, and Dice
ranks the untouched damaged input above every diffusion-restored variant tested at all three seeds
(18/18 configs). SSIM and PSNR each disagree with Dice and with each other about which row is
worst. See `results/milestone2/STAGE7_APPEARANCE_VS_EDGE.md`.

RQ3 *(answered, revised Stage 8)*: does the RQ1 finding replicate at a larger, freshly generated
confirmatory scale (5,000/500/1,000 split), rather than only the 700-diagram pilot used for Gate
G2? **Yes, and it strengthens.** On an independently generated dataset (fresh seed), U0/U1 reach
0.994 mean F1 (sd 0.001–0.002) vs. D0/D1's 0.926 (sd ~0.002) — the gap widens from ~0.05 to ~0.069
F1 and the gap-to-noise ratio rises from ~7–9× to ~29×. The RQ2 divergence also strengthens (Dice
compression ~4.5× → ~6–7.5×). See `results/milestone3/STAGE8_CONFIRMATORY_REPORT.md`.

RQ4 *(answered for three shifts, revised Stage 9)*: does the RQ1 finding generalize to layout,
corruption, or renderer shifts the models were never trained or tuned on? **Yes, for all three
shifts tested.** Models trained once on confirmatory-v1 (3 seeds), evaluated zero-shot against 2×
corruption severity, always-5-node density, and a second renderer
(`geometry.render_antialiased`): the U0/U1-vs-D0/D1 gap holds in every condition (gap-to-noise
ratio ~3.9×–30×) and *strengthens* under density and renderer shift — the U-Net reaches
near-ceiling F1 while diffusion's seed variance rises 14× under the renderer shift specifically.
Only acquisition shift (independently authored/scanned diagrams) remains untested, needing
external human input — see `results/milestone4/STAGE9_GENERALIZATION_REPORT.md` and
`results/milestone4/STAGE9_RENDERER_SHIFT_REPORT.md`.

For every model, plot the U-Net reference as well as diffusion comparisons.

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

A deterministic transformer control uses the same backbone, damaged observation, a zero second channel, fixed time embedding, and a direct restoration objective. It tests whether transformer capacity, rather than iterative denoising, explains any U-Net gap. **Result (Gate G2):** T0 (0.947 F1) underperformed U0/U1 (0.963/0.965) — transformer capacity does not explain a U-Net advantage, because there was no advantage for T0 to explain; if anything this weakly favors the U-Net's convolutional inductive bias for this task. A follow-up 2.9×-larger DiT (D0-tuned) also underperformed its smaller counterpart, reinforcing that more capacity alone does not close the gap. See [results/milestone2/G2_REPORT.md](../results/milestone2/G2_REPORT.md).

## 5. Oracle diagnostic before a learned router — *dropped 13 September 2026, not built*

**Status: not pursued.** Gate G2 showed dense diffusion trailing the U-Net on both accuracy and
latency, confirmed across seeds and four debugging interventions — the premise this stage depends
on (diffusion is worth accelerating) did not hold, and the plan's own failure-response rule says
not to build a complex mechanism to rescue a failed premise. The section below is preserved as the
original design record, not as work performed.

Start with the trained dense backbone. First inspect several equal-token masks without retraining as a sensitivity probe. Because unfamiliar masking can itself degrade a dense model, confirm the finding on a **single shared mask-tolerant checkpoint** fine-tuned with random/varied keep masks. All diagnostic policies use the same weights, examples, timesteps, sampling noise, and executed keep counts.

At keep ratios 0.50 and 0.75, compare random masks (several random draws), regular spatial masks, observation-edge ranking, all-foreground oracle ranking, all-connector oracle ranking, and connector-gap/attachment oracle ranking. If there are more priority patches than the budget, subsample deterministically under the same rule; if fewer, fill from remaining patches with the same fallback rule.

Ground-truth connector/gap locations are allowed **only** in oracle diagnostics or training targets. These are privileged-location heuristics, not guaranteed optimal allocation oracles. Their comparison establishes whether this location information helps under the chosen intervention; it cannot prove a learnable router will match them. Report operation budgets, not deployable oracle speedup.

Proceed when a consistent validation advantage is visible across multiple diagram types and masks; a provisional useful effect is at least 2 F1 points over random at one nontrivial budget. The comparison with all-foreground/observation-edge ranking is necessary to decide whether connection-specific information adds value.

## 6. Smallest proposed router — *dropped 13 September 2026, not built*

**Status: not pursued**, for the same reason as section 5. No router, gate scores, or sparse
gather/scatter inference exist in the codebase. Preserved below as the original design record.

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

## 7. Comparison matrix and fair attribution (revised: B0–D1 are the adopted benchmark; G0–C1 dropped)

All variants receive the same image-only inference information and use the same evaluation pipeline. “Structural loss” below means the same reconstruction-side soft-clDice term.

| ID | Model/policy | Structural loss | Status | What it tests |
| --- | --- | --- | --- | --- |
| B0 | Damaged input unchanged | None | **Run** — `results/milestone2/baselines.json` | Corruption severity and evaluator behavior |
| B1 | Small morphological closing/denoising baseline | None | **Run** | Whether simple repair already solves the task; kernel tuned on validation |
| U0 | Small U-Net | No | **Run** | Required conventional restoration reference |
| U1 | Same U-Net | Yes | **Run** | Whether cheap structural supervision is sufficient |
| T0 | Deterministic restoration transformer | Yes | **Run** | One-pass, similar-backbone alternative to diffusion |
| D0 | Full conditional DiT, 10/20/50 steps | No | **Run**, plus EMA/LR-schedule and capacity variants | Dense quality and fewer-step acceleration |
| D1 | Full conditional DiT, 10/20/50 steps | Yes | **Run** | Benefit of structural loss alone |
| G0 | Learned generic MLP routing | No | *Dropped* | Benefit of ordinary adaptive computation |
| G1 | Learned generic MLP routing | Yes | *Dropped* | Closest matched control for the routing supervision |
| E0 | Observation-edge-ranked fixed routing | Yes | *Dropped* | Cheap importance heuristic available at inference |
| F0 | Learned MLP routing | Yes | *Dropped* | Whether generic foreground supervision explains the result |
| C0 | Complete proposed routing | Yes | *Dropped* | Full routing proposal |
| C1 | Proposed routing without structural restoration loss | No | *Dropped* | Whether the skeleton loss is necessary |

B0–D1 is now the complete, adopted comparison for the benchmark paper; every row has been run at
three seeds against the pilot split (`results/milestone2/`), and D0 has additionally been retrained
with EMA + a warmup-cosine LR schedule and at 2.9× model capacity to rule out fixable training
defects before accepting the D0/D1-vs-U0/U1 gap as architectural. G0–C1 required the dropped
oracle/router (sections 5–6) and were never built; they are kept here only as the historical
record of the original design.

Give U-Net enough development effort to avoid an undertrained straw-man baseline. Report training
wall time and hyperparameter-search cost per row.

Report MACs/FLOPs with an explicit multiply-add convention where useful for the paper; budget
matching for practical claims uses measured end-to-end latency (section 8), which is already
implemented and recorded for every row above.

## 8. Timing, memory, and real operation skipping

**Batch-one latency is implemented** in `diagram_restore.latency.measure_latency` and recorded for
every row in `results/milestone2/baselines.json` (20-warmup/50-repeat screening pass, not yet the
full 5-session protocol below — that rigor is a Stage 8/9 confirmatory-scale task). The
"router/top-k/gather/scatter" and "operation-skipping" material below is specific to the dropped
routing method; the general latency-measurement discipline (warm-up, MPS sync, median/p95/IQR) is
still the live standard for every measurement in this project.

Measure batch-one latency as the primary deployment scenario and a fixed small-batch throughput separately. Use eval/inference mode; identical resolution, precision, device placement, and sampler; fixed power settings; and an idle plugged-in machine. Randomize the order of method timing to reduce thermal/order bias.

Warm up each configuration with at least 20 complete restores. Then time 100 images in at least five repeated sessions, synchronizing MPS immediately before and after timed work. Report median, interquartile range, p95, and uncertainty for latency ratios. Include conditioning, initial noise creation, all denoising steps, router/top-k/gather/scatter, and conversion to the returned image. Report a separate device-only denoising time if useful. Exclude file decoding and graph extraction from the primary restore time, and report them separately. [MPS synchronization](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html)

Instrument MLP input sizes and dense attention costs. Demonstrate that inactive tokens are absent from MLP matrix multiplies. All-keep sparse execution must match the dense output within a validated float32 tolerance; fixed sparse masks must match the dense masked reference. Include any CPU transfers and fallback operations in the measured path.

Log process RSS plus MPS current allocated and driver allocated memory at supported checkpoints. These are different quantities and not additive dedicated-GPU memory. A sampled maximum is a **sampled high-water mark**, not an exact device peak. If precise peak instrumentation is unavailable, say so. Include model, optimizer, and activation costs in training budget reports.

Amdahl-style feasibility estimate: if fraction f of total inference time is the MLP work actually eligible for routing, keep fraction r gives an optimistic time ratio (1-f) + rf, before overhead. For example, f = 0.40 and r = 0.50 yields 0.80 before routing overhead; these numbers are illustrative, not measured. Profile f rather than assuming 50% token removal means 50% faster inference.

## 9. Generalization and confirmatory analysis

This section is unaffected by the pivot and is now the primary remaining empirical work (revised
Stages 8–9 in [docs/03_PROJECT_PLAN.md](03_PROJECT_PLAN.md)), scoped to the retained B0–D1
comparison rather than a routing frontier. Target an expanded 5,000/500/1,000 clean-parent split at 64 × 64, with pilot development parents confined to training/development and a fresh reserved test set. This is a resource-dependent starting size, not a publication minimum or a statistical power guarantee. Decide final test size using development variance and the predeclared effect size before opening test predictions.

Add separate held-out sets for unseen layout families, thinner lines, stronger yet interpretable corruptions, and crowded near-misses. For 128 × 128, profile a matched model first and keep the resolution/capacity change explicit. Add higher-degree nodes and marked free-standing junctions only with an extended tested evaluator: junction dots become graph vertices, and edges terminate at shapes/dots. Never silently treat crossings as junctions or compare different graph conventions in one pooled metric.

Target 50–100 independently authored diagrams rendered by a second tool or drawn/scanned by collaborators, with paired references and manually checked direct edges. Record origin, consent/license, acquisition, registration, and exclusions. A second synthetic renderer tests renderer shift; independently scanned diagrams test an additional acquisition shift. Do not label synthetic re-rendering as real-world validation. Existing diagram datasets require inspection of both license and edge/pair annotations before use.

Freeze model settings, keep ratios, sampling counts, evaluator thresholds, seeds, and primary endpoints on validation. Select latency-matched competitors from validation timing, then evaluate the locked choices on test. No best-of-many diffusion samples: primary output uses a fixed noise seed per image; secondary stochastic analysis averages over predetermined noise seeds without oracle selection.

Report mean and standard deviation across independently trained seeds. Use paired bootstrap intervals over **clean parents**, retaining all corruptions per parent; show each seed's paired effect. For uncertainty spanning optimization variability, also use a hierarchical seed/parent analysis and acknowledge that three seeds give limited seed-level precision. Timing repetitions are not independent model-training replicates. Inspect subgroup harm and extraction failures; do not select only favorable conditions after seeing test results.

## 10. Decision criteria (revised for the benchmark contribution)

Criteria A/B below were routing-vs-dense latency/F1 tradeoff thresholds and no longer apply — there
is no routing method to hold to a tradeoff threshold. The revised criteria for a benchmark paper:

**Criterion C (architecture comparison, met at pilot and confirmatory scale):** the accuracy/
latency ranking among B0/B1/U0/U1/T0/D0/D1 is consistent across at least three training seeds,
with the paired seed-to-seed standard deviation clearly smaller than the gap between the
top-ranked and diffusion-based rows. **Met at pilot scale**: U0/U1 (mean 0.966/0.969 F1) vs. D0/D1
(mean 0.917/0.919 F1) across seeds 7/17/27, a gap roughly 7–9× the seed-to-seed standard
deviation. **Met more strongly at confirmatory scale** (fresh 5,000/500/1,000 dataset): U0/U1
mean 0.994 F1 vs. D0/D1 mean 0.926, a gap roughly 29× the seed-to-seed standard deviation. See
`results/milestone3/STAGE8_CONFIRMATORY_REPORT.md`.

**Criterion D (evaluator-divergence, met at pilot and confirmatory scale):** at least one pair of
restorations exists where an appearance-similarity metric and the direct-edge evaluator disagree
on which is better, or rank the gap between them very differently in magnitude. **Met at pilot
scale** (seeds 7/17/27): mean Dice ranks the untouched damaged input above every
diffusion-restored variant (18/18 configs), compressing the meaningful accuracy range by
4.48× ± 0.60. **Met more strongly at confirmatory scale**: compression widens to ~6–7.5×, and the
damaged input beats diffusion on Dice in 18/18 further configs (36/36 total across both scales).
See `results/milestone2/STAGE7_APPEARANCE_VS_EDGE.md` and
`results/milestone3/STAGE8_CONFIRMATORY_REPORT.md`.

**Criterion E (generalization, met for three shifts — Stage 9):** the Criterion C ranking holds —
or any change is reported honestly — under at least one genuine distribution shift, not just the
exact pilot generator. **Met**: 3 seeds, models trained once on confirmatory-v1, evaluated
zero-shot against 2× corruption severity, always-5-node density, and a second renderer
(`geometry.render_antialiased`) — the ranking holds in all three (gap-to-noise ratio ~3.9×–30×),
strengthening under density and renderer shift. Only an acquisition shift (independently
authored/scanned diagrams) remains open, needing external human input this project cannot
fabricate. See `results/milestone4/STAGE9_GENERALIZATION_REPORT.md` and
`results/milestone4/STAGE9_RENDERER_SHIFT_REPORT.md`.

Do not call a statistically nonsignificant quality difference “equivalent.” These criteria may be
revised with a written development-only justification before the confirmatory-scale (Stage 8) test
set is opened, never retrofitted to final test outcomes.
