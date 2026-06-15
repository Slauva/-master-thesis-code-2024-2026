import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from experiments.logistic_regression.artifacts import (
    summarize_evaluation_runs,
)
from experiments.logistic_regression.config import (
    load_logistic_regression_config,
    parse_dotted_overrides,
)
from experiments.logistic_regression.workflow import execute_evaluation_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logistic-regression",
        description="Train and evaluate random-imagery Logistic Regression experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Train one evaluation protocol.")
    run_parser.add_argument(
        "--protocol",
        required=True,
        choices=("cross-subject", "within-subject"),
    )
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
        help="Validate and reuse all expected immutable runs without fitting.",
    )
    run_parser.add_argument("--json", action="store_true", dest="json_output")
    run_parser.set_defaults(handler=_run_command)

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Safely evaluate persisted metadata and arrays.",
    )
    evaluate_parser.add_argument("run_dirs", nargs="+", type=Path)
    evaluate_parser.add_argument("--json", action="store_true", dest="json_output")
    evaluate_parser.set_defaults(handler=_evaluate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, FileExistsError, PermissionError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run_command(args: argparse.Namespace) -> int:
    overrides = parse_dotted_overrides(args.overrides)
    config = load_logistic_regression_config(
        config_path=args.config,
        overrides=overrides,
    )
    result = execute_evaluation_protocol(
        protocol=args.protocol,
        config=config,
        reuse_existing=args.reuse_existing,
    )
    _print_payload(
        result.summary,
        json_output=args.json_output,
        reused=result.reused,
    )
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    payload = summarize_evaluation_runs(list(args.run_dirs))
    _print_payload(payload, json_output=args.json_output)
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
        _print_run_summary(run)
    combined = payload.get("combined")
    if combined is not None:
        print("Combined within-subject evaluation")
        _print_metrics("  Model", combined["model_metrics"])
        for baseline in combined["baselines"]:
            _print_metrics(f"  Baseline {baseline['name']}", baseline["metrics"])


def _print_run_summary(run: dict[str, Any]) -> None:
    split = run["split"]
    print(f"Run: {run['run_dir']}")
    print(
        f"  Protocol: {run['protocol']} | Direction: {run['direction']['name']}"
    )
    print(
        "  Split: "
        f"{split['n_train_rows']} train rows / {split['n_test_rows']} test rows; "
        f"{len(split['train_subjects'])} train subjects / "
        f"{len(split['test_subjects'])} test subjects"
    )
    print(
        "  Selected feature family: "
        + "+".join(run["selected_feature_family"])
    )
    _print_metrics("  Model", run["model_metrics"])
    for baseline in run["baselines"]:
        _print_metrics(f"  Baseline {baseline['name']}", baseline["metrics"])


def _print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(
        f"{label}: balanced_accuracy={metrics['mean_balanced_accuracy']:.6f}, "
        f"mean_sample_iou={metrics['mean_sample_iou']:.6f}, "
        f"micro_iou={metrics['micro_iou']:.6f}, "
        f"hamming_loss={metrics['hamming_loss']:.6f}"
    )


__all__ = ["build_parser", "main"]
