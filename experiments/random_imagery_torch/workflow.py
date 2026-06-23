from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from experiments.random_imagery.shared import (
    EvaluationProtocol,
    build_evaluation_protocol,
    build_random_imagery_targets,
)
from experiments.random_imagery_torch.artifacts import (
    LoadedTorchRun,
    build_torch_direction_result,
    load_torch_run,
    summarize_torch_runs,
    torch_run_dir,
    write_torch_direction_run,
)
from experiments.random_imagery_torch.config import TorchExperimentConfig
from experiments.random_imagery_torch.models import build_spectral_model
from experiments.random_imagery_torch.spectral_dataset import CropSpectralDataset
from experiments.random_imagery_torch.training import (
    ModelFactory,
    fit_torch_ensemble,
    predict_torch_ensemble,
)


@dataclass(frozen=True, slots=True)
class TorchProtocolWorkflowResult:
    model_id: str
    protocol: EvaluationProtocol
    run_dirs: tuple[Path, ...]
    runs: tuple[LoadedTorchRun, ...]
    summary: dict[str, Any]
    reused: bool


def execute_torch_protocol(
    protocol: EvaluationProtocol,
    *,
    config: TorchExperimentConfig,
    reuse_existing: bool,
    spectral_dataset: Any | None = None,
    targets: Any | None = None,
    model_factory: ModelFactory = build_spectral_model,
) -> TorchProtocolWorkflowResult:
    _validate_artifact_config(config)
    resolved_dataset, resolved_targets = _resolve_inputs(
        config,
        spectral_dataset=spectral_dataset,
        targets=targets,
    )
    definition = build_evaluation_protocol(
        resolved_targets,
        protocol=protocol,
        split_config=config.split,
    )
    expected_run_dirs = tuple(
        torch_run_dir(
            config,
            protocol=definition.protocol,
            direction=direction.name,
        )
        for direction in definition.directions
    )
    existing = tuple(run_dir.is_dir() for run_dir in expected_run_dirs)
    if any(existing):
        if not reuse_existing:
            existing_path = expected_run_dirs[existing.index(True)]
            raise FileExistsError(f"Torch run already exists and is immutable: {existing_path}")
        if not all(existing):
            missing = [
                str(run_dir)
                for run_dir, exists in zip(expected_run_dirs, existing, strict=True)
                if not exists
            ]
            raise FileNotFoundError(f"Cannot reuse an incomplete Torch protocol run set; missing={missing}")
        runs = tuple(load_torch_run(run_dir) for run_dir in expected_run_dirs)
        spectral_hash = getattr(resolved_dataset, "config_hash", None)
        for run in runs:
            if run.config != config:
                raise ValueError(f"Torch configuration differs for reused run: {run.run_dir}")
            if spectral_hash is not None and run.preprocessing["spectral_config_hash"] != spectral_hash:
                raise ValueError(f"Spectral configuration hash differs for reused run: {run.run_dir}")
        return TorchProtocolWorkflowResult(
            model_id=config.model_id,
            protocol=protocol,
            run_dirs=expected_run_dirs,
            runs=runs,
            summary=summarize_torch_runs(list(expected_run_dirs)),
            reused=True,
        )

    run_dirs = []
    for direction, audit in zip(definition.directions, definition.audits, strict=True):
        started = perf_counter()
        fitted = fit_torch_ensemble(
            resolved_dataset,
            resolved_targets,
            direction,
            config=config.training,
            model_factory=model_factory,
        )
        trained_at = perf_counter()
        prediction = predict_torch_ensemble(
            fitted,
            resolved_dataset,
            resolved_targets,
            direction.test_indices,
            config=config.training,
            model_factory=model_factory,
        )
        predicted_at = perf_counter()
        result = build_torch_direction_result(
            direction=direction,
            audit=audit,
            fitted=fitted,
            prediction=prediction,
            targets=resolved_targets,
            config=config,
            training_seconds=trained_at - started,
            prediction_seconds=predicted_at - trained_at,
        )
        run_dirs.append(
            write_torch_direction_run(
                result,
                targets=resolved_targets,
                config=config,
                spectral_config_hash=str(getattr(resolved_dataset, "config_hash", "")),
            )
        )
    run_dir_tuple = tuple(run_dirs)
    return TorchProtocolWorkflowResult(
        model_id=config.model_id,
        protocol=protocol,
        run_dirs=run_dir_tuple,
        runs=tuple(load_torch_run(run_dir) for run_dir in run_dir_tuple),
        summary=summarize_torch_runs(list(run_dir_tuple)),
        reused=False,
    )


def _resolve_inputs(
    config: TorchExperimentConfig,
    *,
    spectral_dataset: Any | None,
    targets: Any | None,
) -> tuple[Any, Any]:
    if (spectral_dataset is None) != (targets is None):
        raise ValueError("Pass both `spectral_dataset` and `targets`, or let the workflow build both")
    if spectral_dataset is not None and targets is not None:
        return spectral_dataset, targets

    from utils.datasets import NumpyDataset

    source_dataset = NumpyDataset(
        config.dataset.dataset_dir,
        dataset_step_type=config.dataset.recording_family,
        dataset_pattern_type=config.dataset.pattern_type,
        cache_policy="disk",
    )
    configured_dataset = CropSpectralDataset(
        source_dataset,
        method=config.training.method,
        preprocessing_config_overrides=config.preprocessing_overrides,
        input_config=config.spectral_input,
        cache_policy="disk",
    )
    configured_targets = build_random_imagery_targets(
        configured_dataset.samples,
        image_rows=config.dataset.image_rows,
        image_columns=config.dataset.image_columns,
        allowed_sample_types=config.dataset.target_sample_types,
    )
    return configured_dataset, configured_targets


def _validate_artifact_config(config: TorchExperimentConfig) -> None:
    if config.artifacts.schema_version != 1:
        raise ValueError("Torch protocol workflow requires artifacts.schema_version=1")
    if config.artifacts.overwrite:
        raise ValueError("Torch protocol workflow does not permit destructive overwrite")


__all__ = ["TorchProtocolWorkflowResult", "execute_torch_protocol"]
