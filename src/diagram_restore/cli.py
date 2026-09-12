import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Diagram restoration research commands")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Generate a new deterministic pilot dataset")
    gen.add_argument("--config", type=Path, default=Path("configs/pilot.toml"))
    gen.add_argument("--output", type=Path, default=Path("data/pilot-v1"))
    audit_parser = commands.add_parser(
        "audit", help="Fixtures, clean integrity and development reports"
    )
    audit_parser.add_argument("--data", type=Path, default=Path("data/pilot-v1"))
    audit_parser.add_argument("--output", type=Path, default=Path("results/milestone1"))
    bench = commands.add_parser("benchmark", help="Bounded training and sparse-MLP benchmarks")
    bench.add_argument("--data", type=Path, default=Path("data/pilot-v1"))
    bench.add_argument("--output", type=Path, default=Path("results/milestone1"))
    bench.add_argument("--steps", type=int, default=200)
    bench.add_argument("--warmup", type=int, default=20)
    overfit = commands.add_parser(
        "tiny-overfit", help="Tiny-set overfit diagnostic before full baseline training"
    )
    overfit.add_argument("--data", type=Path, default=Path("data/pilot-v1"))
    overfit.add_argument("--output", type=Path, default=Path("results/milestone2"))
    overfit.add_argument("--count", type=int, default=12)
    overfit.add_argument("--unet-steps", type=int, default=3000)
    overfit.add_argument("--dit-steps", type=int, default=10000)
    overfit.add_argument("--sampling-steps", type=int, default=50)
    overfit.add_argument("--seed", type=int, default=7)
    baselines = commands.add_parser(
        "baselines", help="B0/B1/U0/U1/T0/D0/D1 baseline suite (Stage 6, Gate G2)"
    )
    baselines.add_argument("--data", type=Path, default=Path("data/pilot-v1"))
    baselines.add_argument("--output", type=Path, default=Path("results/milestone2"))
    baselines.add_argument("--unet-steps", type=int, default=6000)
    baselines.add_argument("--dit-steps", type=int, default=10000)
    baselines.add_argument("--sampling-steps", type=int, nargs="+", default=[10, 20, 50])
    baselines.add_argument("--seed", type=int, default=7)
    baselines.add_argument("--morphological-radii", type=int, nargs="+", default=[0, 1, 2, 3])
    baselines.add_argument("--structural-weight", type=float, default=0.1)
    baselines.add_argument("--latency-warmup", type=int, default=20)
    baselines.add_argument("--latency-repeats", type=int, default=50)
    tuning = commands.add_parser(
        "diffusion-tuning",
        help="EMA + warmup-cosine diffusion training, comparable to the baselines D0 row",
    )
    tuning.add_argument("--data", type=Path, default=Path("data/pilot-v1"))
    tuning.add_argument("--output", type=Path, default=Path("results/milestone2"))
    tuning.add_argument("--steps", type=int, default=30000)
    tuning.add_argument("--sampling-steps", type=int, nargs="+", default=[10, 20, 50])
    tuning.add_argument("--seed", type=int, default=7)
    tuning.add_argument("--structural-weight", type=float, default=0.0)
    tuning.add_argument("--base-lr", type=float, default=1e-4)
    tuning.add_argument("--warmup-steps", type=int, default=1500)
    tuning.add_argument("--ema-decay", type=float, default=0.999)
    tuning.add_argument("--latency-warmup", type=int, default=20)
    tuning.add_argument("--latency-repeats", type=int, default=50)
    args = parser.parse_args()
    if args.command == "generate":
        from .data import generate

        result = generate(args.config, args.output)
    elif args.command == "benchmark":
        from .runtime import benchmark_sparse_mlp, benchmark_training

        result = benchmark_training(args.data, args.output, args.steps, args.warmup)
        result["sparse_mlp"] = benchmark_sparse_mlp(args.output)
    elif args.command == "tiny-overfit":
        from .train import overfit_tiny_set

        result = overfit_tiny_set(
            args.data,
            args.output,
            args.count,
            args.unet_steps,
            args.dit_steps,
            args.sampling_steps,
            args.seed,
        )
    elif args.command == "baselines":
        from .baselines import run_baseline_suite

        result = run_baseline_suite(
            args.data,
            args.output,
            args.unet_steps,
            args.dit_steps,
            tuple(args.sampling_steps),
            args.seed,
            tuple(args.morphological_radii),
            args.structural_weight,
            args.latency_warmup,
            args.latency_repeats,
        )
    elif args.command == "diffusion-tuning":
        from .diffusion_tuning import evaluate_diffusion_tuning

        result = evaluate_diffusion_tuning(
            args.data,
            args.output,
            args.steps,
            tuple(args.sampling_steps),
            args.seed,
            args.structural_weight,
            args.base_lr,
            args.warmup_steps,
            args.ema_decay,
            args.latency_warmup,
            args.latency_repeats,
        )
    else:
        from .artifacts import (
            audit,
            contact_sheets,
            evaluate_observations,
            fixture_sheet,
            overlay_audit,
        )

        result = audit(args.data, args.output)
        if not result["integrity_passed"]:
            print(json.dumps(result, indent=2))
            raise SystemExit(1)
        observations = evaluate_observations(args.data, args.output)
        result["validation_damaged_input"] = observations["overall"]
        result["contact_sheets"] = contact_sheets(args.data, args.output)
        result["fixture_sheet"] = fixture_sheet(args.output)
        result["overlay_audit_count"] = overlay_audit(args.data, args.output)["count"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
