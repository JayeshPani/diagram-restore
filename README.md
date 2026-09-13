# A Direct-Edge Benchmark for Diagram Connectivity Restoration

Research project · prepared 12 September 2026 · rescoped 13 September 2026

**Research question:** Does image-similarity evaluation hide connectivity failures in diagram restoration, and which restoration architectures actually preserve direct connections under a controlled, reproducible corruption benchmark?

This project restores damaged black-and-white diagrams containing boxes, circles, and undirected connectors, and evaluates restorations by their **direct-edge graph**, not just pixel similarity. It was originally scoped around a connectivity-aware diffusion-routing method; Stage 6 experiments (three seeds, four independent debugging interventions) showed a plain U-Net beating dense diffusion on both accuracy and latency, which triggered this plan's own pre-registered failure-response rule. The project is now scoped around the parts that result validated regardless: the corruption generator, the direct-edge evaluator, and a rigorous comparative benchmark. See [docs/01_RESEARCH_SCOPE.md](docs/01_RESEARCH_SCOPE.md)'s amendment and [results/milestone2/G2_REPORT.md](results/milestone2/G2_REPORT.md) for the full evidence trail.

This is document image analysis / graphics recognition, not NLP or text mining, despite the folder name — see docs/01 for that caveat.

## Read these in order

| Document | What it settles |
| --- | --- |
| [Research scope](docs/01_RESEARCH_SCOPE.md) | Problem, method, exclusions, success criteria — v0.2, with the routing-to-benchmark amendment |
| [Literature review](docs/02_LITERATURE_REVIEW.md) | Closest work for the *original* routing framing, plus a two-pass refresh (Stage 12) for the benchmark framing — full-text read of the closest match |
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
- **Stage 7 done, confirmed at 3 seeds** (13 September 2026): appearance metrics hide the connectivity gap. Mean Dice's range is 4.48× ± 0.60 narrower than edge-F1's across seeds 7/17/27, and Dice ranks the untouched damaged input above every diffusion-restored variant tested (18/18 configs). See `results/milestone2/STAGE7_APPEARANCE_VS_EDGE.md`.
- **Literature refresh, two passes done** (13-14 September 2026): closest prior art is SciFlow-Bench (arXiv:2602.09809), read in full text — a near-concurrent independent argument for graph-aware over pixel-based evaluation in diagram *generation*, which also independently found diffusion trailing on structural fidelity (same qualitative pattern as this project's headline, different task). OPRB's venue/authors verified (DocRevive, CVPR 2026 Workshop MULA). See `docs/02_LITERATURE_REVIEW.md`.
- **Stage 8 done** (13 September 2026): both headline findings replicate on a fresh, independently generated 5,000/500/1,000 dataset (3 seeds) — and get *stronger*. U0/U1 reach 0.994 mean F1 vs. D0/D1's 0.926 (gap widens from ~0.05 to ~0.069 F1; gap-to-noise ratio ~7–9× → ~29×), and the Dice/edge-F1 divergence widens from ~4.5× to ~6–7.5× compression, with the damaged input beating every diffusion config on Dice again (36/36 checked configs total, both scales). See `results/milestone3/STAGE8_CONFIRMATORY_REPORT.md`.
- **Stage 9 done for two shifts** (13 September 2026): models trained once per seed (3 seeds) on the confirmatory data, evaluated zero-shot against 2x corruption severity and always-5-node density — the U0/U1-vs-D0/D1 gap holds in every condition (gap-to-noise ratio ~6.6x-30x, no ranking flip). See `results/milestone4/STAGE9_GENERALIZATION_REPORT.md`.
- **Open next steps**: an acquisition shift specifically — real/independently authored or scanned diagrams (the renderer-shift half is now covered, see below); a systematic citation-graph check to close out Stage 12; optionally, additional baseline architectures (second U-Net width, plain CNN autoencoder) a reviewer might expect.
- Accuracy/latency claims are now supported at two scales (pilot: 700 diagrams; confirmatory: 6,500 diagrams), 3 seeds each, on two independently generated in-distribution datasets, and hold under two tested out-of-distribution shifts (corruption severity, layout density). Only a renderer/acquisition shift remains untested.

## Implemented layout

```text
Text Mining/
  README.md
  docs/                        # scope, literature, plan, protocol, budget, paper/venue, search log
  configs/pilot.toml           # pilot dataset generator config (700 diagrams)
  configs/confirmatory.toml    # confirmatory-scale generator config (6,500 diagrams, fresh seed)
  configs/ood_corruption.toml  # Stage 9: 2x noise/blur severity, longer gaps, fresh seed
  configs/ood_density.toml     # Stage 9: always 5 nodes instead of 2-5, fresh seed
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
    generalization.py            # Stage 9: train once, evaluate against held-out OOD shifts
    runtime.py                   # environment inventory, training/sparse-MLP benchmarks
    artifacts.py                 # audit reports, contact sheets, overlays
    cli.py                       # `diagram-restore generate|audit|benchmark|tiny-overfit|baselines|diffusion-tuning|generalization`
  tests/                        # 167 tests: evaluator fixtures, sampler correctness, training smoke tests
  data/pilot-v1/                # pilot dataset (700 diagrams); excluded from Git
  data/confirmatory-v1/         # confirmatory dataset (6,500 diagrams, fresh seed); excluded from Git
  data/ood-corruption-v1/       # Stage 9 OOD test set (harder corruption); excluded from Git
  data/ood-density-v1/          # Stage 9 OOD test set (denser layouts); excluded from Git
  results/
    milestone1/                 # Gate G1: environment, benchmark, integrity, fixtures, contact sheets
    milestone2/                 # Gate G2 + Stage 7: baselines.json, G2_REPORT.md, STAGE7_APPEARANCE_VS_EDGE.md, diffusion-tuning ablations, seed17/, seed27/
    milestone3/                 # Stage 8: confirmatory-scale integrity audit, seed7/, seed17/, seed27/, STAGE8_CONFIRMATORY_REPORT.md
    milestone4/                 # Stage 9: seed7/, seed17/, seed27/, STAGE9_GENERALIZATION_REPORT.md
  paper/                        # manuscript and figures, after the remaining lit refresh and mentor review
```

Run `diagram-restore --help` (after `uv sync`) for the available commands. Each stage ends with a result to inspect and a written decision — these are scientific checkpoints, not just implementation milestones.
