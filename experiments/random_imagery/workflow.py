from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.random_imagery.artifacts import (
    LoadedModelRun,
    load_model_run,
    model_run_dir,
    summarize_model_runs,
    write_model_protocol_runs,
)
from experiments.random_imagery.backends import build_model_backend
from experiments.random_imagery.config import RandomImageryModelConfig
from experiments.random_imagery.runner import run_model_evaluation_protocol
from experiments.random_imagery.shared import (
    EvaluationProtocol,
    build_evaluation_protocol,
    build_random_imagery_targets,
)


@dataclass(frozen=True, slots=True)
class ModelProtocolWorkflowResult:
    model_id: str
    protocol: EvaluationProtocol
    run_dirs: tuple[Path, ...]
    runs: tuple[LoadedModelRun, ...]
    summary: dict[str, Any]
    reused: bool


def execute_model_protocol(
    protocol: EvaluationProtocol,
    *,
    config: RandomImageryModelConfig,
    reuse_existing: bool,
    dataset: Any | None = None,
    targets: Any | None = None,
) -> ModelProtocolWorkflowResult:
    _validate_artifact_config(config)
    resolved_dataset, resolved_targets = _resolve_inputs(
        config,
        dataset=dataset,
        targets=targets,
    )
    definition = build_evaluation_protocol(
        resolved_targets,
        protocol=protocol,
        split_config=config.split,
    )
    expected_run_dirs = tuple(
        model_run_dir(
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
            raise FileExistsError(
                f"Experiment run already exists and is immutable: {existing_path}"
            )
        if not all(existing):
            missing = [
                str(run_dir)
                for run_dir, exists in zip(expected_run_dirs, existing, strict=True)
                if not exists
            ]
            raise FileNotFoundError(
                f"Cannot reuse an incomplete protocol run set; missing={missing}"
            )
        runs = tuple(load_model_run(run_dir) for run_dir in expected_run_dirs)
        for run in runs:
            if run.features["feature_config_hash"] != resolved_dataset.config_hash:
                raise ValueError(
                    f"Feature configuration hash differs for reused run: {run.run_dir}"
                )
            if run.config != config:
                raise ValueError(f"Configuration differs for reused run: {run.run_dir}")
        return ModelProtocolWorkflowResult(
            model_id=config.model_id,
            protocol=protocol,
            run_dirs=expected_run_dirs,
            runs=runs,
            summary=summarize_model_runs(list(expected_run_dirs)),
            reused=True,
        )

    result = run_model_evaluation_protocol(
        protocol,
        config=config,
        backend=build_model_backend(config.model_id),
        dataset=resolved_dataset,
        targets=resolved_targets,
    )
    run_dirs = write_model_protocol_runs(
        result,
        targets=resolved_targets,
        config=config,
        feature_config_hash=resolved_dataset.config_hash,
    )
    return ModelProtocolWorkflowResult(
        model_id=config.model_id,
        protocol=protocol,
        run_dirs=run_dirs,
        runs=tuple(load_model_run(run_dir) for run_dir in run_dirs),
        summary=summarize_model_runs(list(run_dirs)),
        reused=False,
    )


def _resolve_inputs(
    config: RandomImageryModelConfig,
    *,
    dataset: Any | None,
    targets: Any | None,
) -> tuple[Any, Any]:
    if (dataset is None) != (targets is None):
        raise ValueError("Pass both `dataset` and `targets`, or let the workflow build both")
    if dataset is not None and targets is not None:
        return dataset, targets

    from utils.datasets import FeatureDataset

    configured_dataset = FeatureDataset(
        config.dataset.dataset_dir,
        dataset_step_type=config.dataset.recording_family,
        dataset_pattern_type=config.dataset.pattern_type,
        config_path=config.dataset.feature_config_path,
        cache_policy="disk",
        source_cache_policy="disk",
    )
    configured_targets = build_random_imagery_targets(
        configured_dataset.samples,
        image_rows=config.dataset.image_rows,
        image_columns=config.dataset.image_columns,
        allowed_sample_types=config.dataset.target_sample_types,
    )
    return configured_dataset, configured_targets


def _validate_artifact_config(config: RandomImageryModelConfig) -> None:
    if config.artifacts.schema_version != 3:
        raise ValueError("Model protocol workflow requires artifacts.schema_version=3")
    if config.artifacts.overwrite:
        raise ValueError("Model protocol workflow does not permit destructive overwrite")


__all__ = ["ModelProtocolWorkflowResult", "execute_model_protocol"]
