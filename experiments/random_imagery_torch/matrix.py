import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

from omegaconf import OmegaConf

from experiments.random_imagery.shared import build_random_imagery_targets
from experiments.random_imagery_torch.config import (
    PRIMARY_TORCH_MODEL_IDS,
    PreprocessingMethod,
    load_torch_config,
    parse_torch_model_id,
)
from experiments.random_imagery_torch.models import build_spectral_model
from experiments.random_imagery_torch.training import ModelFactory
from experiments.random_imagery_torch.workflow import execute_torch_protocol

TorchMatrixProtocol = Literal["cross-subject", "within-subject"]

TORCH_MATRIX_PROTOCOLS: tuple[TorchMatrixProtocol, ...] = (
    "cross-subject",
    "within-subject",
)
FULL_IMAGERY_TORCH_ARTIFACT_ROOT = Path("artifacts/experiments/full-imagery/torch")
FULL_IMAGERY_TORCH_SUMMARY_PATH = Path(
    "artifacts/experiments/full-imagery/stage4_torch_matrix_summary.json"
)
FULL_IMAGERY_TORCH_FAILURES_PATH = Path(
    "artifacts/experiments/full-imagery/stage4_torch_matrix_failures.json"
)


@dataclass(frozen=True, slots=True)
class TorchMatrixRunSpec:
    model_id: str
    protocol: TorchMatrixProtocol
    artifact_root: Path = FULL_IMAGERY_TORCH_ARTIFACT_ROOT

    def __post_init__(self) -> None:
        if self.model_id not in PRIMARY_TORCH_MODEL_IDS:
            supported = ", ".join(PRIMARY_TORCH_MODEL_IDS)
            raise ValueError(f"Unsupported Torch matrix model {self.model_id!r}; expected one of: {supported}")
        if self.protocol not in TORCH_MATRIX_PROTOCOLS:
            supported = ", ".join(TORCH_MATRIX_PROTOCOLS)
            raise ValueError(f"Unsupported protocol {self.protocol!r}; expected one of: {supported}")

    @property
    def architecture(self) -> str:
        architecture, _method = parse_torch_model_id(self.model_id)
        return architecture

    @property
    def method(self) -> PreprocessingMethod:
        _architecture, method = parse_torch_model_id(self.model_id)
        return method

    @property
    def expected_direction_runs(self) -> int:
        return 1 if self.protocol == "cross-subject" else 2

    @property
    def overrides(self) -> dict[str, Any]:
        return {
            "dataset": {"pattern_type": None},
            "artifacts": {"root": self.artifact_root.as_posix()},
        }

    @property
    def dotted_overrides(self) -> tuple[str, ...]:
        return (
            "dataset.pattern_type=null",
            f"artifacts.root={self.artifact_root.as_posix()}",
        )

    @property
    def command(self) -> tuple[str, ...]:
        command = [
            "random-imagery-torch",
            "run",
            "--model",
            self.model_id,
            "--protocol",
            self.protocol,
        ]
        for override in self.dotted_overrides:
            command.extend(("--set", override))
        return tuple(command)

    @property
    def plan_id(self) -> str:
        payload = {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "method": self.method,
            "protocol": self.protocol,
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
            "architecture": self.architecture,
            "method": self.method,
            "protocol": self.protocol,
            "command": list(self.command),
            "overrides": self.overrides,
            "dotted_overrides": list(self.dotted_overrides),
            "expected_direction_runs": self.expected_direction_runs,
        }


def build_torch_matrix_plan(
    *,
    model_ids: Sequence[str] = PRIMARY_TORCH_MODEL_IDS,
    protocols: Sequence[TorchMatrixProtocol] = TORCH_MATRIX_PROTOCOLS,
    artifact_root: Path = FULL_IMAGERY_TORCH_ARTIFACT_ROOT,
) -> tuple[TorchMatrixRunSpec, ...]:
    return tuple(
        TorchMatrixRunSpec(
            model_id=model_id,
            protocol=protocol,
            artifact_root=Path(artifact_root),
        )
        for model_id in model_ids
        for protocol in protocols
    )


def build_torch_matrix_plan_payload(specs: Sequence[TorchMatrixRunSpec]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_count": len(specs),
        "expected_direction_run_count": sum(spec.expected_direction_runs for spec in specs),
        "models": sorted({spec.model_id for spec in specs}),
        "architectures": sorted({spec.architecture for spec in specs}),
        "methods": sorted({spec.method for spec in specs}),
        "protocols": sorted({spec.protocol for spec in specs}),
        "runs": [spec.to_jsonable() for spec in specs],
    }


def execute_torch_matrix_sweep(
    *,
    specs: Sequence[TorchMatrixRunSpec] | None = None,
    reuse_existing: bool = True,
    continue_on_error: bool = True,
    output_path: Path = FULL_IMAGERY_TORCH_SUMMARY_PATH,
    failure_log_path: Path = FULL_IMAGERY_TORCH_FAILURES_PATH,
    extra_overrides: dict[str, Any] | None = None,
    spectral_dataset: Any | None = None,
    targets: Any | None = None,
    model_factory: ModelFactory = build_spectral_model,
    verbose: bool = False,
) -> dict[str, Any]:
    resolved_specs = tuple(specs) if specs is not None else build_torch_matrix_plan()
    if not resolved_specs:
        raise ValueError("At least one Torch matrix spec is required")
    if (spectral_dataset is None) != (targets is None):
        raise ValueError("Pass both `spectral_dataset` and `targets`, or let the sweep build both")
    if spectral_dataset is not None and len({spec.method for spec in resolved_specs}) != 1:
        raise ValueError("A shared `spectral_dataset` can only be used for one preprocessing method")

    started_at = datetime.now(UTC).isoformat()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    plan_payload = build_torch_matrix_plan_payload(resolved_specs)
    input_cache: dict[PreprocessingMethod, tuple[Any, Any]] = {}

    for index, spec in enumerate(resolved_specs, start=1):
        start = time.perf_counter()
        try:
            resolved_dataset, resolved_targets = (
                (spectral_dataset, targets)
                if spectral_dataset is not None and targets is not None
                else _cached_full_imagery_inputs(
                    spec,
                    cache=input_cache,
                    extra_overrides=extra_overrides,
                )
            )
            result = _execute_torch_matrix_spec(
                spec,
                reuse_existing=reuse_existing,
                extra_overrides=extra_overrides,
                spectral_dataset=resolved_dataset,
                targets=resolved_targets,
                model_factory=model_factory,
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
                    "summary": _compact_torch_protocol_summary(result["summary"]),
                }
            )
            if verbose:
                print(
                    f"[{index}/{len(resolved_specs)}] completed "
                    f"{spec.model_id} {spec.protocol} "
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
                    f"{spec.model_id} {spec.protocol}: {type(error).__name__}: {error}",
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


def _execute_torch_matrix_spec(
    spec: TorchMatrixRunSpec,
    *,
    reuse_existing: bool,
    extra_overrides: dict[str, Any] | None,
    spectral_dataset: Any,
    targets: Any,
    model_factory: ModelFactory,
) -> dict[str, Any]:
    config = load_torch_config(
        spec.model_id,
        overrides=_merged_overrides(extra_overrides, spec.overrides),
    )
    result = execute_torch_protocol(
        spec.protocol,
        config=config,
        reuse_existing=reuse_existing,
        spectral_dataset=spectral_dataset,
        targets=targets,
        model_factory=model_factory,
    )
    return {
        "reused": result.reused,
        "run_dirs": [path.as_posix() for path in result.run_dirs],
        "summary": result.summary,
    }


def _cached_full_imagery_inputs(
    spec: TorchMatrixRunSpec,
    *,
    cache: dict[PreprocessingMethod, tuple[Any, Any]],
    extra_overrides: dict[str, Any] | None,
) -> tuple[Any, Any]:
    if spec.method not in cache:
        cache[spec.method] = _resolve_full_imagery_torch_inputs(
            spec,
            extra_overrides=extra_overrides,
        )
    return cache[spec.method]


def _resolve_full_imagery_torch_inputs(
    spec: TorchMatrixRunSpec,
    *,
    extra_overrides: dict[str, Any] | None,
) -> tuple[Any, Any]:
    config = load_torch_config(
        spec.model_id,
        overrides=_merged_overrides(extra_overrides, spec.overrides),
    )
    from experiments.random_imagery_torch.spectral_dataset import CropSpectralDataset
    from utils.datasets import NumpyDataset

    source_dataset = NumpyDataset(
        config.dataset.dataset_dir,
        dataset_step_type=config.dataset.recording_family,
        dataset_pattern_type=config.dataset.pattern_type,
        cache_policy="disk",
    )
    spectral_dataset = CropSpectralDataset(
        source_dataset,
        method=config.training.method,
        preprocessing_config_overrides=config.preprocessing_overrides,
        input_config=config.spectral_input,
        cache_policy="disk",
    )
    resolved_targets = build_random_imagery_targets(
        spectral_dataset.samples,
        image_rows=config.dataset.image_rows,
        image_columns=config.dataset.image_columns,
        allowed_sample_types=config.dataset.target_sample_types,
    )
    return spectral_dataset, resolved_targets


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
        raise TypeError("Merged Torch matrix overrides must be a mapping")
    return payload


def _compact_torch_protocol_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "runs": [_compact_torch_run_summary(run) for run in summary.get("runs", [])],
        "combined": (
            _compact_torch_combined_summary(summary["combined"])
            if summary.get("combined") is not None
            else None
        ),
    }


def _compact_torch_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    architecture, method = parse_torch_model_id(run["model_id"])
    return {
        "run_dir": run["run_dir"],
        "artifact_schema_version": run["artifact_schema_version"],
        "model_id": run["model_id"],
        "architecture": architecture,
        "method": method,
        "protocol": run["protocol"],
        "direction": run["direction"]["name"],
        "score_semantics": run.get("score_semantics", "native_probability"),
        "mean_balanced_accuracy": run["model_metrics"]["mean_balanced_accuracy"],
        "mean_macro_f1": run["model_metrics"]["mean_macro_f1"],
        "mean_score_mse": run["model_metrics"]["mean_score_mse"],
        "bit_accuracy": run["model_metrics"]["bit_accuracy"],
        "exact_match_accuracy": run["model_metrics"]["exact_match_accuracy"],
        "hamming_loss": run["model_metrics"]["hamming_loss"],
    }


def _compact_torch_combined_summary(combined: dict[str, Any]) -> dict[str, Any]:
    architecture, method = parse_torch_model_id(combined["model_id"])
    return {
        "model_id": combined["model_id"],
        "architecture": architecture,
        "method": method,
        "protocol": combined["protocol"],
        "direction_names": combined["direction_names"],
        "score_semantics": combined.get("score_semantics", "native_probability"),
        "mean_balanced_accuracy": combined["model_metrics"]["mean_balanced_accuracy"],
        "mean_macro_f1": combined["model_metrics"]["mean_macro_f1"],
        "mean_score_mse": combined["model_metrics"]["mean_score_mse"],
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
        "architectures": plan_payload["architectures"],
        "methods": plan_payload["methods"],
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
    "FULL_IMAGERY_TORCH_ARTIFACT_ROOT",
    "FULL_IMAGERY_TORCH_FAILURES_PATH",
    "FULL_IMAGERY_TORCH_SUMMARY_PATH",
    "TORCH_MATRIX_PROTOCOLS",
    "TorchMatrixProtocol",
    "TorchMatrixRunSpec",
    "build_torch_matrix_plan",
    "build_torch_matrix_plan_payload",
    "execute_torch_matrix_sweep",
]
