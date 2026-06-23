from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.logistic_regression.artifacts import (
    LoadedEvaluationRun,
    evaluation_run_dir,
    load_evaluation_run,
    summarize_evaluation_runs,
    write_protocol_evaluation_runs,
)
from experiments.logistic_regression.config import LogisticRegressionExperimentConfig
from experiments.logistic_regression.data import (
    build_evaluation_protocol,
    build_random_imagery_targets,
)
from experiments.logistic_regression.runner import run_evaluation_protocol
from experiments.logistic_regression.schemas import EvaluationProtocol


@dataclass(frozen=True, slots=True)
class ProtocolWorkflowResult:
    protocol: EvaluationProtocol
    run_dirs: tuple[Path, ...]
    runs: tuple[LoadedEvaluationRun, ...]
    summary: dict[str, Any]
    reused: bool


def execute_evaluation_protocol(
    protocol: EvaluationProtocol,
    *,
    config: LogisticRegressionExperimentConfig,
    reuse_existing: bool,
    dataset: Any | None = None,
    targets: Any | None = None,
) -> ProtocolWorkflowResult:
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
        evaluation_run_dir(
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
        runs = tuple(load_evaluation_run(run_dir) for run_dir in expected_run_dirs)
        for run in runs:
            if run.features["feature_config_hash"] != resolved_dataset.config_hash:
                raise ValueError(
                    f"Feature configuration hash differs for reused run: {run.run_dir}"
                )
        return ProtocolWorkflowResult(
            protocol=protocol,
            run_dirs=expected_run_dirs,
            runs=runs,
            summary=summarize_evaluation_runs(list(expected_run_dirs)),
            reused=True,
        )

    result = run_evaluation_protocol(
        protocol,
        config=config,
        dataset=resolved_dataset,
        targets=resolved_targets,
    )
    run_dirs = write_protocol_evaluation_runs(
        result,
        targets=resolved_targets,
        config=config,
        feature_config_hash=resolved_dataset.config_hash,
    )
    return ProtocolWorkflowResult(
        protocol=protocol,
        run_dirs=run_dirs,
        runs=tuple(load_evaluation_run(run_dir) for run_dir in run_dirs),
        summary=summarize_evaluation_runs(list(run_dirs)),
        reused=False,
    )


def _resolve_inputs(
    config: LogisticRegressionExperimentConfig,
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


def _validate_artifact_config(
    config: LogisticRegressionExperimentConfig,
) -> None:
    if config.artifacts.schema_version != 2:
        raise ValueError("Protocol workflow requires artifacts.schema_version=2")
    if config.artifacts.overwrite:
        raise ValueError("Protocol workflow does not permit destructive artifact overwrite")


__all__ = ["ProtocolWorkflowResult", "execute_evaluation_protocol"]
