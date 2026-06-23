import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from experiments.random_imagery.config import parse_dotted_overrides
from experiments.random_imagery_torch.artifacts import (
    compare_torch_runs,
    summarize_torch_runs,
)
from experiments.random_imagery_torch.config import (
    PRIMARY_TORCH_MODEL_IDS,
    load_torch_config,
)
from experiments.random_imagery_torch.matrix import (
    FULL_IMAGERY_TORCH_ARTIFACT_ROOT,
    FULL_IMAGERY_TORCH_FAILURES_PATH,
    FULL_IMAGERY_TORCH_SUMMARY_PATH,
    build_torch_matrix_plan,
    build_torch_matrix_plan_payload,
    execute_torch_matrix_sweep,
)
from experiments.random_imagery_torch.workflow import execute_torch_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="random-imagery-torch",
        description="Train, evaluate, and compare Torch spectral random-imagery models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Train one Torch model and protocol.")
    run_parser.add_argument("--model", required=True, choices=PRIMARY_TORCH_MODEL_IDS)
    run_parser.add_argument("--protocol", required=True, choices=("cross-subject", "within-subject"))
    run_parser.add_argument("--config", type=Path)
    run_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted OmegaConf override.",
    )
    run_parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Validate and reuse every expected immutable direction run.",
    )
    run_parser.add_argument("--json", action="store_true", dest="json_output")
    run_parser.set_defaults(handler=_run_command)

    matrix_plan_parser = commands.add_parser(
        "matrix-plan",
        help="Enumerate the full-imagery Torch architecture/method/protocol matrix.",
    )
    matrix_plan_parser.add_argument(
        "--model",
        action="append",
        choices=PRIMARY_TORCH_MODEL_IDS,
        help="Restrict to one or more model ids.",
    )
    matrix_plan_parser.add_argument(
        "--protocol",
        action="append",
        choices=("cross-subject", "within-subject"),
        help="Restrict to one or more protocols.",
    )
    matrix_plan_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=FULL_IMAGERY_TORCH_ARTIFACT_ROOT,
    )
    matrix_plan_parser.add_argument("--json", action="store_true", dest="json_output")
    matrix_plan_parser.set_defaults(handler=_matrix_plan_command)

    matrix_run_parser = commands.add_parser(
        "matrix-run",
        help="Execute the full-imagery Torch architecture/method/protocol matrix.",
    )
    matrix_run_parser.add_argument(
        "--model",
        action="append",
        choices=PRIMARY_TORCH_MODEL_IDS,
        help="Restrict to one or more model ids.",
    )
    matrix_run_parser.add_argument(
        "--protocol",
        action="append",
        choices=("cross-subject", "within-subject"),
        help="Restrict to one or more protocols.",
    )
    matrix_run_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=FULL_IMAGERY_TORCH_ARTIFACT_ROOT,
    )
    matrix_run_parser.add_argument(
        "--summary-output",
        type=Path,
        default=FULL_IMAGERY_TORCH_SUMMARY_PATH,
    )
    matrix_run_parser.add_argument(
        "--failure-log",
        type=Path,
        default=FULL_IMAGERY_TORCH_FAILURES_PATH,
    )
    matrix_run_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted OmegaConf override applied before matrix-required overrides.",
    )
    matrix_run_parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="Refuse existing immutable runs instead of validating and reusing them.",
    )
    matrix_run_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed matrix spec.",
    )
    matrix_run_parser.add_argument("--json", action="store_true", dest="json_output")
    matrix_run_parser.set_defaults(handler=_matrix_run_command)

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Safely evaluate Torch metadata and arrays without loading weights.",
    )
    evaluate_parser.add_argument("run_dirs", nargs="+", type=Path)
    evaluate_parser.add_argument("--json", action="store_true", dest="json_output")
    evaluate_parser.set_defaults(handler=_evaluate_command)

    compare_parser = commands.add_parser(
        "compare",
        help="Compare compatible Torch direction runs.",
    )
    compare_parser.add_argument("run_dirs", nargs="+", type=Path)
    compare_parser.add_argument("--json", action="store_true", dest="json_output")
    compare_parser.set_defaults(handler=_compare_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        FileNotFoundError,
        FileExistsError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run_command(args: argparse.Namespace) -> int:
    config = load_torch_config(
        args.model,
        config_path=args.config,
        overrides=parse_dotted_overrides(args.overrides),
    )
    result = execute_torch_protocol(
        protocol=args.protocol,
        config=config,
        reuse_existing=args.reuse_existing,
    )
    _print_payload(result.summary, json_output=args.json_output, reused=result.reused)
    return 0


def _matrix_plan_command(args: argparse.Namespace) -> int:
    specs = build_torch_matrix_plan(
        model_ids=tuple(args.model) if args.model else PRIMARY_TORCH_MODEL_IDS,
        protocols=tuple(args.protocol) if args.protocol else ("cross-subject", "within-subject"),
        artifact_root=args.artifact_root,
    )
    payload = build_torch_matrix_plan_payload(specs)
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return 0
    print(
        "Planned "
        f"{payload['run_count']} Torch protocol runs / "
        f"{payload['expected_direction_run_count']} direction runs"
    )
    for spec in specs:
        print(" ".join(spec.command))
    return 0


def _matrix_run_command(args: argparse.Namespace) -> int:
    specs = build_torch_matrix_plan(
        model_ids=tuple(args.model) if args.model else PRIMARY_TORCH_MODEL_IDS,
        protocols=tuple(args.protocol) if args.protocol else ("cross-subject", "within-subject"),
        artifact_root=args.artifact_root,
    )
    summary = execute_torch_matrix_sweep(
        specs=specs,
        reuse_existing=not args.no_reuse_existing,
        continue_on_error=not args.fail_fast,
        output_path=args.summary_output,
        failure_log_path=args.failure_log,
        extra_overrides=parse_dotted_overrides(args.overrides),
        verbose=not args.json_output,
    )
    if args.json_output:
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    else:
        print(
            "Completed Torch protocol runs: "
            f"{summary['completed_protocol_run_count']} / "
            f"{summary['planned_protocol_run_count']} | "
            f"failures: {summary['failed_protocol_run_count']}"
        )
        print(f"Summary: {args.summary_output}")
        print(f"Failure log: {args.failure_log}")
    return 1 if summary["failed_protocol_run_count"] else 0


def _evaluate_command(args: argparse.Namespace) -> int:
    _print_payload(summarize_torch_runs(list(args.run_dirs)), json_output=args.json_output)
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    _print_comparison(compare_torch_runs(list(args.run_dirs)), json_output=args.json_output)
    return 0


def _print_payload(
    payload: dict[str, Any],
    *,
    json_output: bool,
    reused: bool | None = None,
) -> None:
    if json_output:
        output = dict(payload)
        if reused is not None:
            output["reused"] = reused
        print(json.dumps(output, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return
    if reused is not None:
        print(f"Mode: {'reused existing runs' if reused else 'trained new runs'}")
    for run in payload["runs"]:
        print(
            f"{run['model_id']} | {run['protocol']} | "
            f"{run['direction']['name']} | "
            f"balanced_accuracy={run['model_metrics']['mean_balanced_accuracy']:.6f}"
        )
    combined = payload.get("combined")
    if combined is not None:
        print(
            f"{combined['model_id']} | combined within-subject | "
            f"balanced_accuracy={combined['model_metrics']['mean_balanced_accuracy']:.6f}"
        )


def _print_comparison(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return
    print(f"Protocol: {payload['protocol']} | Direction: {payload['direction']}")
    for run in payload["runs"]:
        print(
            f"{run['model_id']}: "
            f"balanced_accuracy={run['model_metrics']['mean_balanced_accuracy']:.6f}, "
            f"difference_vs_first={run['balanced_accuracy_difference_vs_first']:+.6f}"
        )


__all__ = ["build_parser", "main"]
