import hashlib
import json
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

from omegaconf import OmegaConf
from scipy.linalg import LinAlgWarning

from experiments.logistic_regression.config import load_logistic_regression_config
from experiments.logistic_regression.workflow import execute_evaluation_protocol
from experiments.random_imagery.config import load_model_config
from experiments.random_imagery.data import build_random_imagery_targets
from experiments.random_imagery.registry import (
    MODEL_REGISTRY,
    PLANNED_MODEL_IDS,
    REFERENCE_MODEL_ID,
)
from experiments.random_imagery.workflow import execute_model_protocol

MatrixProtocol = Literal["cross-subject", "within-subject"]
MatrixRunner = Literal["logistic-regression", "random-imagery-models"]

TABULAR_FEATURE_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("time",),
    ("spectral",),
    ("time", "spectral"),
    ("covariance",),
    ("correlation",),
    ("log_covariance",),
    ("lndp",),
    ("lgp",),
    ("lbp",),
)
CLASSICAL_MATRIX_MODEL_IDS: tuple[str, ...] = (
    REFERENCE_MODEL_ID,
    *PLANNED_MODEL_IDS,
)
MATRIX_PROTOCOLS: tuple[MatrixProtocol, ...] = ("cross-subject", "within-subject")
FULL_IMAGERY_CLASSICAL_ARTIFACT_ROOT = Path(
    "artifacts/experiments/full-imagery/classical"
)
FULL_IMAGERY_CLASSICAL_SUMMARY_PATH = Path(
    "artifacts/experiments/full-imagery/stage3_classical_matrix_summary.json"
)
FULL_IMAGERY_CLASSICAL_FAILURES_PATH = Path(
    "artifacts/experiments/full-imagery/stage3_classical_matrix_failures.json"
)


@dataclass(frozen=True, slots=True)
class MatrixRunSpec:
    model_id: str
    feature_family: tuple[str, ...]
    protocol: MatrixProtocol
    artifact_root: Path = FULL_IMAGERY_CLASSICAL_ARTIFACT_ROOT

    def __post_init__(self) -> None:
        if self.model_id not in MODEL_REGISTRY:
            supported = ", ".join(CLASSICAL_MATRIX_MODEL_IDS)
            raise ValueError(f"Unsupported matrix model {self.model_id!r}; expected one of: {supported}")
        if self.feature_family not in TABULAR_FEATURE_FAMILIES:
            supported = ", ".join(feature_family_slug(item) for item in TABULAR_FEATURE_FAMILIES)
            raise ValueError(
                f"Unsupported feature family {feature_family_slug(self.feature_family)!r}; "
                f"expected one of: {supported}"
            )
        if self.protocol not in MATRIX_PROTOCOLS:
            supported = ", ".join(MATRIX_PROTOCOLS)
            raise ValueError(f"Unsupported protocol {self.protocol!r}; expected one of: {supported}")

    @property
    def feature_slug(self) -> str:
        return feature_family_slug(self.feature_family)

    @property
    def runner(self) -> MatrixRunner:
        if self.model_id == REFERENCE_MODEL_ID:
            return "logistic-regression"
        return "random-imagery-models"

    @property
    def run_root(self) -> Path:
        return self.artifact_root / self.model_id / self.feature_slug

    @property
    def expected_direction_runs(self) -> int:
        return 1 if self.protocol == "cross-subject" else 2

    @property
    def overrides(self) -> dict[str, Any]:
        return {
            "dataset": {"pattern_type": None},
            "feature_screening": {"candidates": [list(self.feature_family)]},
            "artifacts": {"root": self.run_root.as_posix()},
        }

    @property
    def dotted_overrides(self) -> tuple[str, ...]:
        return (
            "dataset.pattern_type=null",
            f"feature_screening.candidates={_feature_candidate_dotlist(self.feature_family)}",
            f"artifacts.root={self.run_root.as_posix()}",
        )

    @property
    def command(self) -> tuple[str, ...]:
        command = [self.runner, "run"]
        if self.runner == "random-imagery-models":
            command.extend(("--model", self.model_id))
        command.extend(("--protocol", self.protocol))
        for override in self.dotted_overrides:
            command.extend(("--set", override))
        return tuple(command)

    @property
    def plan_id(self) -> str:
        payload = {
            "model_id": self.model_id,
            "feature_family": self.feature_family,
            "protocol": self.protocol,
            "runner": self.runner,
            "overrides": self.overrides,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hashlib.sha256(canonical).hexdigest()[:16]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "model_id": self.model_id,
            "feature_family": list(self.feature_family),
            "feature_slug": self.feature_slug,
            "protocol": self.protocol,
            "runner": self.runner,
            "command": list(self.command),
            "overrides": self.overrides,
            "dotted_overrides": list(self.dotted_overrides),
            "expected_direction_runs": self.expected_direction_runs,
        }


def build_classical_matrix_plan(
    *,
    model_ids: Sequence[str] = CLASSICAL_MATRIX_MODEL_IDS,
    feature_families: Sequence[tuple[str, ...]] = TABULAR_FEATURE_FAMILIES,
    protocols: Sequence[MatrixProtocol] = MATRIX_PROTOCOLS,
    artifact_root: Path = FULL_IMAGERY_CLASSICAL_ARTIFACT_ROOT,
) -> tuple[MatrixRunSpec, ...]:
    return tuple(
        MatrixRunSpec(
            model_id=model_id,
            feature_family=tuple(feature_family),
            protocol=protocol,
            artifact_root=Path(artifact_root),
        )
        for model_id in model_ids
        for feature_family in feature_families
        for protocol in protocols
    )


def build_matrix_plan_payload(specs: Sequence[MatrixRunSpec]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_count": len(specs),
        "expected_direction_run_count": sum(spec.expected_direction_runs for spec in specs),
        "models": sorted({spec.model_id for spec in specs}),
        "feature_families": sorted({spec.feature_slug for spec in specs}),
        "protocols": sorted({spec.protocol for spec in specs}),
        "runs": [spec.to_jsonable() for spec in specs],
    }


def execute_classical_matrix_sweep(
    *,
    specs: Sequence[MatrixRunSpec] | None = None,
    reuse_existing: bool = True,
    continue_on_error: bool = True,
    output_path: Path = FULL_IMAGERY_CLASSICAL_SUMMARY_PATH,
    failure_log_path: Path = FULL_IMAGERY_CLASSICAL_FAILURES_PATH,
    extra_overrides: dict[str, Any] | None = None,
    dataset: Any | None = None,
    targets: Any | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    resolved_specs = tuple(specs) if specs is not None else build_classical_matrix_plan()
    if not resolved_specs:
        raise ValueError("At least one matrix spec is required")
    if (dataset is None) != (targets is None):
        raise ValueError("Pass both `dataset` and `targets`, or let the sweep build both")
    resolved_dataset, resolved_targets = (
        _resolve_full_imagery_inputs(resolved_specs[0], extra_overrides=extra_overrides)
        if dataset is None and targets is None
        else (dataset, targets)
    )
    started_at = datetime.now(UTC).isoformat()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    plan_payload = build_matrix_plan_payload(resolved_specs)

    for index, spec in enumerate(resolved_specs, start=1):
        start = time.perf_counter()
        try:
            result = _execute_matrix_spec(
                spec,
                reuse_existing=reuse_existing,
                extra_overrides=extra_overrides,
                dataset=resolved_dataset,
                targets=resolved_targets,
            )
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "status": "completed",
                    "index": index,
                    "duration_seconds": elapsed,
                    **spec.to_jsonable(),
                    "reused": result["reused"],
                    "run_dirs": result["run_dirs"],
                    "summary": _compact_protocol_summary(result["summary"]),
                }
            )
            if verbose:
                print(
                    f"[{index}/{len(resolved_specs)}] completed "
                    f"{spec.model_id} {spec.feature_slug} {spec.protocol} "
                    f"reused={result['reused']} elapsed={elapsed:.1f}s",
                    flush=True,
                )
        except Exception as error:
            elapsed = time.perf_counter() - start
            failure = {
                "status": "failed",
                "index": index,
                "duration_seconds": elapsed,
                **spec.to_jsonable(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            results.append(failure)
            if verbose:
                print(
                    f"[{index}/{len(resolved_specs)}] failed "
                    f"{spec.model_id} {spec.feature_slug} {spec.protocol}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            if not continue_on_error:
                summary = _sweep_summary_payload(
                    plan_payload=plan_payload,
                    started_at=started_at,
                    results=results,
                    failures=failures,
                    complete=False,
                )
                _write_json_atomic(output_path, summary)
                _write_json_atomic(failure_log_path, {"failures": failures})
                raise
        summary = _sweep_summary_payload(
            plan_payload=plan_payload,
            started_at=started_at,
            results=results,
            failures=failures,
            complete=len(results) == len(resolved_specs),
        )
        _write_json_atomic(output_path, summary)
        _write_json_atomic(failure_log_path, {"failures": failures})

    return _sweep_summary_payload(
        plan_payload=plan_payload,
        started_at=started_at,
        results=results,
        failures=failures,
        complete=True,
    )


def feature_family_slug(feature_family: Sequence[str]) -> str:
    if not feature_family:
        raise ValueError("Feature family must not be empty")
    return "+".join(feature_family)


def feature_family_from_slug(slug: str) -> tuple[str, ...]:
    for feature_family in TABULAR_FEATURE_FAMILIES:
        if feature_family_slug(feature_family) == slug:
            return feature_family
    supported = ", ".join(feature_family_slug(item) for item in TABULAR_FEATURE_FAMILIES)
    raise ValueError(f"Unsupported feature family slug {slug!r}; expected one of: {supported}")


def _feature_candidate_dotlist(feature_family: tuple[str, ...]) -> str:
    return "[[" + ",".join(feature_family) + "]]"


def _execute_matrix_spec(
    spec: MatrixRunSpec,
    *,
    reuse_existing: bool,
    extra_overrides: dict[str, Any] | None,
    dataset: Any,
    targets: Any,
) -> dict[str, Any]:
    overrides = _merged_overrides(extra_overrides, spec.overrides)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        if spec.model_id == REFERENCE_MODEL_ID:
            config = load_logistic_regression_config(overrides=overrides)
            result = execute_evaluation_protocol(
                spec.protocol,
                config=config,
                reuse_existing=reuse_existing,
                dataset=dataset,
                targets=targets,
            )
        else:
            config = load_model_config(spec.model_id, overrides=overrides)
            result = execute_model_protocol(
                spec.protocol,
                config=config,
                reuse_existing=reuse_existing,
                dataset=dataset,
                targets=targets,
            )
    return {
        "reused": result.reused,
        "run_dirs": [path.as_posix() for path in result.run_dirs],
        "summary": result.summary,
    }


def _resolve_full_imagery_inputs(
    spec: MatrixRunSpec,
    *,
    extra_overrides: dict[str, Any] | None,
) -> tuple[Any, Any]:
    overrides = _merged_overrides(extra_overrides, spec.overrides)
    config = (
        load_logistic_regression_config(overrides=overrides)
        if spec.model_id == REFERENCE_MODEL_ID
        else load_model_config(spec.model_id, overrides=overrides)
    )
    from utils.datasets import FeatureDataset

    dataset = FeatureDataset(
        config.dataset.dataset_dir,
        dataset_step_type=config.dataset.recording_family,
        dataset_pattern_type=config.dataset.pattern_type,
        config_path=config.dataset.feature_config_path,
        cache_policy="disk",
        source_cache_policy="disk",
    )
    targets = build_random_imagery_targets(
        dataset.samples,
        image_rows=config.dataset.image_rows,
        image_columns=config.dataset.image_columns,
        allowed_sample_types=config.dataset.target_sample_types,
    )
    return dataset, targets


def _merged_overrides(
    first: dict[str, Any] | None,
    second: dict[str, Any],
) -> dict[str, Any]:
    configs: list[Any] = []
    if first:
        configs.append(OmegaConf.create(first))
    configs.append(OmegaConf.create(second))
    merged = OmegaConf.merge(*configs)
    payload = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise TypeError("Merged matrix overrides must be a mapping")
    return payload


def _compact_protocol_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "runs": [_compact_run_summary(run) for run in summary.get("runs", [])],
        "combined": (
            _compact_combined_summary(summary["combined"])
            if summary.get("combined") is not None
            else None
        ),
    }


def _compact_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": run["run_dir"],
        "artifact_schema_version": run["artifact_schema_version"],
        "model_id": run.get("model_id", REFERENCE_MODEL_ID),
        "protocol": run["protocol"],
        "direction": run["direction"]["name"],
        "selected_feature_family": run["selected_feature_family"],
        "score_semantics": run.get("score_semantics", "native_probability"),
        "mean_balanced_accuracy": run["model_metrics"]["mean_balanced_accuracy"],
        "mean_macro_f1": run["model_metrics"]["mean_macro_f1"],
        "mean_score_mse": run["model_metrics"].get(
            "mean_score_mse",
            run["model_metrics"].get("mean_brier_score"),
        ),
        "bit_accuracy": run["model_metrics"]["bit_accuracy"],
        "exact_match_accuracy": run["model_metrics"]["exact_match_accuracy"],
        "hamming_loss": run["model_metrics"]["hamming_loss"],
    }


def _compact_combined_summary(combined: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": combined.get("model_id", REFERENCE_MODEL_ID),
        "protocol": combined["protocol"],
        "direction_names": combined["direction_names"],
        "score_semantics": combined.get("score_semantics", "native_probability"),
        "mean_balanced_accuracy": combined["model_metrics"]["mean_balanced_accuracy"],
        "mean_macro_f1": combined["model_metrics"]["mean_macro_f1"],
        "mean_score_mse": combined["model_metrics"].get(
            "mean_score_mse",
            combined["model_metrics"].get("mean_brier_score"),
        ),
        "bit_accuracy": combined["model_metrics"]["bit_accuracy"],
        "exact_match_accuracy": combined["model_metrics"]["exact_match_accuracy"],
        "hamming_loss": combined["model_metrics"]["hamming_loss"],
    }


def _sweep_summary_payload(
    *,
    plan_payload: dict[str, Any],
    started_at: str,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    completed = [result for result in results if result["status"] == "completed"]
    return {
        "schema_version": 1,
        "started_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "complete": complete,
        "planned_protocol_run_count": plan_payload["run_count"],
        "planned_direction_run_count": plan_payload["expected_direction_run_count"],
        "completed_protocol_run_count": len(completed),
        "failed_protocol_run_count": len(failures),
        "completed_direction_run_count": sum(
            len(result.get("run_dirs", [])) for result in completed
        ),
        "models": plan_payload["models"],
        "feature_families": plan_payload["feature_families"],
        "protocols": plan_payload["protocols"],
        "results": results,
        "failures": failures,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "CLASSICAL_MATRIX_MODEL_IDS",
    "FULL_IMAGERY_CLASSICAL_ARTIFACT_ROOT",
    "FULL_IMAGERY_CLASSICAL_FAILURES_PATH",
    "FULL_IMAGERY_CLASSICAL_SUMMARY_PATH",
    "MATRIX_PROTOCOLS",
    "TABULAR_FEATURE_FAMILIES",
    "MatrixProtocol",
    "MatrixRunner",
    "MatrixRunSpec",
    "build_classical_matrix_plan",
    "build_matrix_plan_payload",
    "execute_classical_matrix_sweep",
    "feature_family_from_slug",
    "feature_family_slug",
]
