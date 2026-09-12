# A Direct-Edge Benchmark for Diagram Connectivity Restoration

Research project · prepared 12 September 2026 · rescoped 13 September 2026

**Research question:** Does image-similarity evaluation hide connectivity failures in diagram restoration, and which restoration architectures actually preserve direct connections under a controlled, reproducible corruption benchmark?

This project restores damaged black-and-white diagrams containing boxes, circles, and undirected connectors, and evaluates restorations by their **direct-edge graph**, not just pixel similarity. It was originally scoped around a connectivity-aware diffusion-routing method; Stage 6 experiments (three seeds, four independent debugging interventions) showed a plain U-Net beating dense diffusion on both accuracy and latency, which triggered this plan's own pre-registered failure-response rule. The project is now scoped around the parts that result validated regardless: the corruption generator, the direct-edge evaluator, and a rigorous comparative benchmark. See [docs/01_RESEARCH_SCOPE.md](docs/01_RESEARCH_SCOPE.md)'s amendment and [results/milestone2/G2_REPORT.md](results/milestone2/G2_REPORT.md) for the full evidence trail.

This is document image analysis / graphics recognition, not NLP or text mining, despite the folder name — see docs/01 for that caveat.

## Read these in order

| Document | What it settles |
| --- | --- |
| [Research scope](docs/01_RESEARCH_SCOPE.md) | Problem, method, exclusions, success criteria — v0.2, with the routing-to-benchmark amendment |
| [Literature review](docs/02_LITERATURE_REVIEW.md) | Closest work for the *original* routing framing — **needs a refresh** for the benchmark framing before submission (Stage 12) |
| [Staged project plan](docs/03_PROJECT_PLAN.md) | Stages 1–6 complete; Stages 7–12 revised to a benchmark contribution after Gate G2 |
| [Experiment protocol](docs/04_EXPERIMENT_PROTOCOL.md) | Dataset, evaluator, training recipe, and revised decision criteria; sections 5–6 (oracle/router) kept only as a dropped-design record |
| [Environment and budget](docs/05_ENVIRONMENT_AND_BUDGET.md) | Verified Mac/software inventory and compute budget (unaffected by the pivot) |
| [Paper and venue plan](docs/06_PAPER_AND_VENUE.md) | Revised claim-to-evidence checklist, manuscript structure, and ICDAR 2027 candidacy |
| [Research search log](docs/07_SEARCH_LOG.md) | Original search queries; flagged for a benchmark-framing refresh |
| [Gate G2 report](results/milestone2/G2_REPORT.md) | The evidence that triggered the pivot: 3 seeds, 4 debugging interventions, all ruled out as fixable explanations |

## Current status

- **Gate G1 passed** (dataset + evaluator): 700-diagram pilot split (500/100/100), 80/80 hand-specified evaluator fixtures pass, exact clean-graph recovery. See `results/milestone1/`.
- **Gate G2 screened** (baselines): U0/U1 (small U-Net) dominate D0/D1 (dense conditional diffusion) on both direct-edge F1 (0.966–0.969 vs. 0.917–0.919 mean) and latency (2.1 ms vs. 20–160 ms), confirmed across seeds 7/17/27 and after ruling out undertraining, missing training tricks (EMA + LR schedule), and insufficient capacity (2.9× larger model) as explanations. See `results/milestone2/G2_REPORT.md`.
- **Scope pivot** (13 September 2026): the connectivity-aware routing method is dropped; no oracle or learned router was built. The project is rescoped to the benchmark contribution — see docs/01's amendment.
- **Stage 7 done** (13 September 2026, single seed): appearance metrics hide the connectivity gap. Mean Dice spans 0.9875–0.9986 (range 0.0111) vs. edge-F1's 0.9110–0.9652 (range 0.0542, ~5× wider), and Dice ranks the untouched damaged input above every diffusion-restored variant tested. See `results/milestone2/STAGE7_APPEARANCE_VS_EDGE.md`.
- **Open next steps**: confirm the Stage 7 divergence at seeds 17/27; an expanded confirmatory-scale benchmark at 5,000/500/1,000 diagrams with repeated seeds (Stage 8, a multi-hour compute commitment — not yet started), generalization to held-out distributions (Stage 9), and a literature refresh for the benchmark framing (Stage 12).
- No accuracy/latency claim beyond the pilot scale (700 diagrams, 3 seeds) has been made; no confirmatory-scale or generalization result exists yet.

## Implemented layout

```text
Text Mining/
  README.md
  docs/                        # scope, literature, plan, protocol, budget, paper/venue, search log
  configs/pilot.toml           # dataset generator config
  src/diagram_restore/
    data.py                    # deterministic corruption generator, manifest, leakage-safe splits
    geometry.py                # node/connector rasterization shared by the generator and evaluator
    evaluation.py               # direct-edge extraction, edge metrics, appearance metrics
    fixtures.py                 # 80 hand-specified evaluator counterexamples
    models.py                  # SmallUNet, ConditionalDiT, losses (incl. soft-clDice structural loss)
    sampling.py                 # DDIM reverse sampler
    train.py                    # tiny-set overfit diagnostic
    baselines.py                 # B0/B1/U0/U1/T0/D0/D1 baseline suite + batch-one latency
    diffusion_tuning.py          # EMA + warmup-cosine LR + capacity variants for D0 debugging
    latency.py                   # batch-one restoration latency measurement
    runtime.py                   # environment inventory, training/sparse-MLP benchmarks
    artifacts.py                 # audit reports, contact sheets, overlays
    cli.py                       # `diagram-restore generate|audit|benchmark|tiny-overfit|baselines|diffusion-tuning`
  tests/                        # 163 tests: evaluator fixtures, sampler correctness, training smoke tests
  data/pilot-v1/                # generated dataset; excluded from Git
  results/
    milestone1/                 # Gate G1: environment, benchmark, integrity, fixtures, contact sheets
    milestone2/                 # Gate G2: baselines.json, G2_REPORT.md, diffusion-tuning ablations, seed17/, seed27/
  paper/                        # manuscript and figures, after Stage 7/9 evidence exists
```

Run `diagram-restore --help` (after `uv sync`) for the available commands. Each stage ends with a result to inspect and a written decision — these are scientific checkpoints, not just implementation milestones.
