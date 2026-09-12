# Connectivity-Aware Adaptive Diffusion for Diagram Restoration

Research project plan · prepared 12 September 2026

**Research question:** Can connectivity-aware routing reduce the computation of a diffusion model while preserving connections in restored diagrams?

The proposed project restores damaged black-and-white diagrams containing boxes, circles, and undirected connectors. It tests whether a small conditional diffusion transformer can preserve the correct node-to-node connections while selectively skipping feedforward computation.

This is a viable **pilot research project**, with a conditional route to a conference paper. Novelty, useful restoration quality, and practical acceleration still require experiments. The supplied instructions determine the scope; despite the folder name, this is principally document image analysis and graphics recognition, rather than NLP or text mining.

## Read these in order

| Document | What it settles |
| --- | --- |
| [Research scope](docs/01_RESEARCH_SCOPE.md) | One-page problem, method, exclusions, and proposed success criteria |
| [Literature review](docs/02_LITERATURE_REVIEW.md) | Closest work, overlapping ideas, implementation references, and the remaining research opportunity |
| [Staged project plan](docs/03_PROJECT_PLAN.md) | Twelve stages, checkable outputs, decision gates, and a ten-week working schedule |
| [Experiment protocol](docs/04_EXPERIMENT_PROTOCOL.md) | Dataset, graph evaluator, architecture, routing supervision, baselines, timing, and statistical analysis |
| [Environment and budget](docs/05_ENVIRONMENT_AND_BUDGET.md) | Verified Mac/software inventory and how to estimate actual experiment costs |
| [Paper and venue plan](docs/06_PAPER_AND_VENUE.md) | Evidence needed for the paper and a currently verified submission candidate |
| [Research search log](docs/07_SEARCH_LOG.md) | Queries, evidence access, exclusions, and remaining coverage limits |

## Current status

- Complete: supplied instructions read; initial literature and novelty assessment; hardware/software inventory; written research and execution plan.
- Verified: Apple M4 Pro, 24 GB unified memory; installed PyTorch 2.9.0 imports successfully and reports MPS available.
- Pending: isolated project environment, training benchmark, dataset generator, evaluator, model implementations, and experiments.
- No accuracy, runtime, memory benchmark, or publication result is claimed. No long training jobs or paid compute have been started.

## First implementation milestone

Generate 500 training, 100 validation, and 100 reserved test diagrams at 64 × 64. Save each clean/damaged pair and graph annotations. Build an evaluator that correctly identifies intact, missing, and invented **direct** connections on deliberately constructed examples. Produce a contact sheet and a short validation report.

Begin model training only after that measurement system passes its checks. The router comes later, after the baseline and oracle gates.

## Proposed implementation layout

Only the planning documents are implemented at this point. The following application directories are to be added in the stages described in the plan:

```text
Text Mining/
  README.md
  docs/
  configs/                   # dataset, model, training, and evaluation settings
  src/diagram_restore/
    data/                    # graph sampling, rendering, corruption, manifests
    models/                  # U-Net, dense DiT, routed DiT
    training/                # losses, optimizers, checkpoints, seeds
    evaluation/              # direct-edge extraction, image metrics, timing
  scripts/                   # generate, train, evaluate, benchmark, visualize
  tests/                     # evaluator fixtures, leakage checks, sparse parity
  data/                      # generated files; excluded from Git by default
  results/                   # compact reports and metrics tracked; heavy outputs ignored
  paper/                     # manuscript and figures, after evidence exists
```

Each stage ends with a result to inspect and a written decision. These checkpoints are scientific decisions about whether the next stage is justified.
