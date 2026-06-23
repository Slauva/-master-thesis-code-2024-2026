import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

from experiments.random_imagery.artifacts import (
    compare_runs,
    summarize_model_runs,
)
from experiments.random_imagery.config import (
    load_model_config,
    parse_dotted_overrides,
)
from experiments.random_imagery.matrix import (
    CLASSICAL_MATRIX_MODEL_IDS,
    FULL_IMAGERY_CLASSICAL_ARTIFACT_ROOT,
    FULL_IMAGERY_CLASSICAL_FAILURES_PATH,
    FULL_IMAGERY_CLASSICAL_SUMMARY_PATH,
    MATRIX_PROTOCOLS,
    TABULAR_FEATURE_FAMILIES,
    build_classical_matrix_plan,
    build_matrix_plan_payload,
    execute_classical_matrix_sweep,
    feature_family_from_slug,
    feature_family_slug,
)
from experiments.random_imagery.registry import PLANNED_MODEL_IDS
from experiments.random_imagery.workflow import execute_model_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="random-imagery-models",
        description="Train, evaluate, and compare classical random-imagery models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Train one model and protocol.")
    run_parser.add_argument("--model", required=True, choices=PLANNED_MODEL_IDS)
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
        help="Validate and reuse every expected immutable direction run.",
    )
    run_parser.add_argument("--json", action="store_true", dest="json_output")
    run_parser.set_defaults(handler=_run_command)

    matrix_parser = commands.add_parser(
        "matrix-plan",
        help="Enumerate fixed full-imagery classical model-feature runs without training.",
    )
    matrix_parser.add_argument(
        "--protocol",
        action="append",
        choices=MATRIX_PROTOCOLS,
        help="Protocol to include; repeat to select multiple. Defaults to all protocols.",
    )
    matrix_parser.add_argument(
        "--model",
        action="append",
        choices=CLASSICAL_MATRIX_MODEL_IDS,
        help="Model to include; repeat to select multiple. Defaults to all classical models.",
    )
    matrix_parser.add_argument(
        "--feature-family",
        action="append",
        choices=tuple(feature_family_slug(item) for item in TABULAR_FEATURE_FAMILIES),
        help="Feature family to include; repeat to select multiple. Defaults to all tabular families.",
    )
    matrix_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=FULL_IMAGERY_CLASSICAL_ARTIFACT_ROOT,
        help="Root for planned immutable full-imagery classical artifacts.",
    )
    matrix_parser.add_argument("--json", action="store_true", dest="json_output")
    matrix_parser.set_defaults(handler=_matrix_plan_command)

    matrix_run_parser = commands.add_parser(
        "matrix-run",
        help="Execute fixed full-imagery classical model-feature runs.",
    )
    matrix_run_parser.add_argument(
        "--protocol",
        action="append",
        choices=MATRIX_PROTOCOLS,
        help="Protocol to include; repeat to select multiple. Defaults to all protocols.",
    )
    matrix_run_parser.add_argument(
        "--model",
        action="append",
        choices=CLASSICAL_MATRIX_MODEL_IDS,
        help="Model to include; repeat to select multiple. Defaults to all classical models.",
    )
    matrix_run_parser.add_argument(
        "--feature-family",
        action="append",
        choices=tuple(feature_family_slug(item) for item in TABULAR_FEATURE_FAMILIES),
        help="Feature family to include; repeat to select multiple. Defaults to all tabular families.",
    )
    matrix_run_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=FULL_IMAGERY_CLASSICAL_ARTIFACT_ROOT,
        help="Root for planned immutable full-imagery classical artifacts.",
    )
    matrix_run_parser.add_argument(
        "--summary-output",
        type=Path,
        default=FULL_IMAGERY_CLASSICAL_SUMMARY_PATH,
    )
    matrix_run_parser.add_argument(
        "--failure-log",
        type=Path,
        default=FULL_IMAGERY_CLASSICAL_FAILURES_PATH,
    )
    matrix_run_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted OmegaConf override applied before fixed matrix overrides.",
    )
    matrix_run_parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="Fail on existing immutable runs instead of validating and reusing them.",
    )
    matrix_run_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed matrix spec.",
    )
    matrix_run_parser.add_argument("--json", action="store_true", dest="json_output")
    matrix_run_parser.set_defaults(handler=_matrix_run_command)

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Safely evaluate schema-v3 metadata and arrays without joblib.",
    )
    evaluate_parser.add_argument("run_dirs", nargs="+", type=Path)
    evaluate_parser.add_argument("--json", action="store_true", dest="json_output")
    evaluate_parser.set_defaults(handler=_evaluate_command)

    compare_parser = commands.add_parser(
        "compare",
        help="Compare compatible schema-v2 or schema-v3 direction runs.",
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
    except (FileNotFoundError, FileExistsError, PermissionError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run_command(args: argparse.Namespace) -> int:
    config = load_model_config(
        args.model,
        config_path=args.config,
        overrides=parse_dotted_overrides(args.overrides),
    )
    result = execute_model_protocol(
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


def _matrix_plan_command(args: argparse.Namespace) -> int:
    specs = _build_matrix_specs(args)
    payload = build_matrix_plan_payload(specs)
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        return 0

    print(
        f"Planned protocol runs: {payload['run_count']} | "
        f"expected direction runs: {payload['expected_direction_run_count']}"
    )
    for spec in specs:
        print(shlex.join(spec.command))
    return 0


def _matrix_run_command(args: argparse.Namespace) -> int:
    specs = _build_matrix_specs(args)
    summary = execute_classical_matrix_sweep(
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
        return 0
    print(
        f"Completed protocol runs: {summary['completed_protocol_run_count']} / "
        f"{summary['planned_protocol_run_count']} | "
        f"failures: {summary['failed_protocol_run_count']}"
    )
    print(f"Summary: {args.summary_output}")
    print(f"Failure log: {args.failure_log}")
    return 0 if summary["failed_protocol_run_count"] == 0 else 1


def _build_matrix_specs(args: argparse.Namespace) -> tuple[Any, ...]:
    feature_families = (
        tuple(feature_family_from_slug(slug) for slug in args.feature_family)
        if args.feature_family
        else TABULAR_FEATURE_FAMILIES
    )
    return build_classical_matrix_plan(
        model_ids=tuple(args.model) if args.model else CLASSICAL_MATRIX_MODEL_IDS,
        feature_families=feature_families,
        protocols=tuple(args.protocol) if args.protocol else MATRIX_PROTOCOLS,
        artifact_root=args.artifact_root,
    )


def _evaluate_command(args: argparse.Namespace) -> int:
    _print_payload(
        summarize_model_runs(list(args.run_dirs)),
        json_output=args.json_output,
    )
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    _print_comparison(
        compare_runs(list(args.run_dirs)),
        json_output=args.json_output,
    )
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
