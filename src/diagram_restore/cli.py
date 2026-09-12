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
