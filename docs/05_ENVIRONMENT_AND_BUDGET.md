# Environment and experiment budget

Inspected locally on 12 September 2026. This report separates verified inventory from proposed setup and unmeasured performance.

## Verified inventory

| Item | Observed value |
| --- | --- |
| Project directory | `/Users/jayeshpani/Projects/Text Mining` |
| Computer | MacBook Pro, model Mac16,8 |
| Chip | Apple M4 Pro |
| CPU cores | 14: 10 performance, 4 efficiency |
| Unified memory | 24 GB |
| Architecture | arm64 |
| Operating system | macOS 26.3.1, build 25D2128 |
| Default Python | 3.14.0 at `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3` |
| Available project Python candidate | 3.12.12 at `/opt/homebrew/bin/python3.12` |
| Imported default-interpreter PyTorch | 2.9.0 |
| MPS built / available | True / True |
| CUDA available | False |
| Git | 2.50.1, Apple Git-155 |
| uv | 0.10.7 |
| Available disk space at inspection | Approximately 109 GiB; recheck before expansion |

Default-interpreter packages also included NumPy 2.2.6, Pillow 12.3.0, SciPy 1.16.3, NetworkX 3.6, PyYAML 6.0.3, and psutil 7.1.3. scikit-image and pytest were absent from that interpreter. These observations are not a validated project dependency lock.

The directory was empty and outside any Git repository at inspection. The planning task initializes a local Git repository for the authored files; no remote publication is needed for planning.

MPS availability establishes that PyTorch can detect the backend. It does not prove every chosen operation is supported, that training fits memory, or that sparse routing is faster. [Official MPS guide](https://docs.pytorch.org/docs/stable/notes/mps.html)

## Proposed isolated setup

Use Python 3.12.12 for the project so its environment is independent of the default interpreter. Start with PyTorch 2.9.0 as the locally verified import baseline, then verify it again inside the environment. Resolve NumPy, Pillow, SciPy, scikit-image, NetworkX, PyYAML, matplotlib, psutil, and pytest there, and freeze the exact successful versions. A newer PyTorch is an experiment decision, not a necessary first step.

Add a `pyproject.toml` and reproducible dependency lock during Stage 3. Save `sys.version`, OS/device inventory, imported package versions, selected backend, precision, environment variables affecting MPS, and Git revision in each experiment record. Do not claim bitwise MPS determinism without testing it.

Use MPS for model execution and CPU for generation/graph extraction. First check forward/backward, attention, top-k, gather/scatter, normalization, and soft-skeleton operations. Report any CPU fallback explicitly. A shared CPU reference on tiny tensors helps diagnose numerical differences. Current “stable” web documentation redirected to 2.14, while the inspected local package is 2.9.0; verify API availability in the locked environment.

## Benchmark before a long run

Benchmark the actual implemented U-Net and compact DiT with their intended losses and optimizer. Use 20 warm-up training steps, then 200 timed steps at batch sizes 4 and 8. Synchronize around timed MPS work and measure data loading separately. Record median/p95 step time, process RSS, MPS allocator/driver observations, actual parameter count, device transfers, and whether swap pressure rises.

Then benchmark one dense versus sparse MLP block at the intended token/feature dimensions and keep fractions. Follow with the full inference path once the sampler exists. An isolated block gain is not an end-to-end acceleration result.

Initial operating targets: comfortably fit within 24 GB unified memory, aim for measured process/device usage consistent with roughly a 12 GB project working-set allowance, and avoid sustained swapping. RSS and MPS memory overlap and must not be blindly added. Keep the first full diffusion run below a provisional six-hour cap. If it does not fit, reduce batch size/width/depth and recheck learning capacity before considering larger resources.

## Cost worksheet

For each run, estimate:

```text
training_hours = measured_seconds_per_step * planned_optimizer_steps / 3600
evaluation_hours = number_of_restores * measured_seconds_per_restore / 3600
total_hours = training_hours + evaluation_hours + checkpoint/setup overhead
```

Validation diffusion sampling can be substantial. Budget it explicitly; use a small fixed development subset for frequent checks and the full validation set at predetermined checkpoints. Count inherited pretraining and failed/tuning runs in total research cost.

Illustration for 20,000 optimizer steps, excluding evaluation and other overhead:

| Assumed seconds/step | Illustrative training time |
| --- | --- |
| 0.05 | 0.28 hours |
| 0.20 | 1.11 hours |
| 0.50 | 2.78 hours |
| 1.00 | 5.56 hours |

**None of these rates has been measured on this project.** They show how the short benchmark determines feasibility. The actual optimizer-step budget is chosen after the benchmark and learning-curve inspection; one training step samples a diffusion timestep, not an entire 1,000-step reverse trajectory.

## Provisional local compute ceilings

| Phase | Device-time allowance | Purpose |
| --- | --- | --- |
| G1/G2 development | Up to 24 hours | Benchmarks, tiny-set debugging, first baselines and sampling sweep |
| G3/G4 development | Up to 24 hours | Oracle, sparse execution, one-seed router/control screening |
| Confirmatory training | Up to 48 hours | Fresh/seed-specific pretraining and matched fine-tuning for the retained comparison set |
| Final evaluation | Up to 16 hours | Held-out/OOD sampling, repeated timing, extraction and variability reports |
| Total planning ceiling | 112 hours | Conditional ceiling, not expected consumption or a launched workload |

This ceiling may not fund every proposed comparison if measured costs are high. Resolve that before final protocol freeze: prioritize essential controls, narrow the claim, or revise the schedule/resources. Do not spend a large budget on a router before G2/G3, omit controls silently, or assume cloud access. No paid computing service is authorized or provisioned by this plan.

Retain compact metrics and selected checkpoints; reserve a provisional 15 GB project data/results budget initially. At the current disk availability, the small pilot is a reasonable storage target, but expanded predictions and repeated checkpoints need explicit retention planning. Do not delete source data or failed-run records merely to tidy results.

## Pending measured fields

Environment lock: pending. U-Net/DiT parameters: pending. Training seconds/step: pending. Full restoration latency: pending. Sparse fraction of runtime: pending. Peak or sampled memory: pending. Cost of all required comparisons: pending. MPS operator compatibility beyond backend detection: pending.
