import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from experiments.random_imagery.baselines import build_non_eeg_baselines
from experiments.random_imagery.config import ArtifactConfig
from experiments.random_imagery.metrics import (
    PredictionMetrics,
    SubjectBootstrapInterval,
    bootstrap_subject_mean_balanced_accuracy,
    evaluate_prediction_matrix,
)
from experiments.random_imagery.schemas import (
    BaselinePrediction,
    EvaluationDirection,
    PixelTargetDataset,
    ProtocolLeakageAudit,
)
from experiments.random_imagery_torch.config import (
    TorchExperimentConfig,
    build_torch_run_hash,
)
from experiments.random_imagery_torch.models import (
    SpectralModelShape,
    build_spectral_model,
)
from experiments.random_imagery_torch.schemas import SpectralNormalizationState
from experiments.random_imagery_torch.spectral_dataset import CropSpectralDataset
from experiments.random_imagery_torch.training import ModelFactory
from experiments.random_imagery_torch.training_schemas import (
    EnsembleMember,
    FittedTorchEnsemble,
    ModelCheckpointState,
    TorchEnsemblePrediction,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TorchBaselineEvaluation:
    prediction: BaselinePrediction
    metrics: PredictionMetrics


@dataclass(frozen=True, slots=True)
class TorchDirectionEvaluationResult:
    direction: EvaluationDirection
    audit: ProtocolLeakageAudit
    fitted: FittedTorchEnsemble
    prediction: TorchEnsemblePrediction
    model_metrics: PredictionMetrics
    model_bootstrap: SubjectBootstrapInterval
    baselines: tuple[TorchBaselineEvaluation, ...]
    training_seconds: float | None = None
    prediction_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.audit.direction_name != self.direction.name:
            raise ValueError("Torch direction result audit does not match its direction")
        if self.audit.has_forbidden_leakage:
            raise ValueError("Torch direction result contains forbidden leakage")
        if not np.array_equal(
            self.fitted.training_target_indices,
            self.direction.train_indices,
        ):
            raise ValueError("Torch fitted rows do not match direction training rows")
        if not np.array_equal(
            self.prediction.test_target_indices,
            self.direction.test_indices,
        ):
            raise ValueError("Torch prediction rows do not match direction test rows")
        if tuple(item.prediction.name for item in self.baselines) != (
            "global_majority",
            "pixel_frequency",
            "seeded_bernoulli",
        ):
            raise ValueError("Torch direction result must contain the canonical baselines")


@dataclass(frozen=True, slots=True)
class LoadedTorchRun:
    run_dir: Path
    manifest: dict[str, Any]
    config: TorchExperimentConfig
    environment: dict[str, Any]
    split: dict[str, Any]
    preprocessing: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    scores: NDArray[np.float64]
    predictions: NDArray[np.int8]
    member_scores: NDArray[np.float64]
    test_balanced_accuracy: NDArray[np.float64]
    train_targets: NDArray[np.int8]
    test_targets: NDArray[np.int8]
    test_subject_ids: NDArray[np.int64]
    baseline_scores: dict[str, NDArray[np.float64]]
    baseline_predictions: dict[str, NDArray[np.int8]]
    normalization: SpectralNormalizationState
    checkpoints: tuple[ModelCheckpointState, ...] = ()

    def __post_init__(self) -> None:
        _validate_loaded_arrays(
            scores=self.scores,
            predictions=self.predictions,
            member_scores=self.member_scores,
            test_balanced_accuracy=self.test_balanced_accuracy,
            train_targets=self.train_targets,
            test_targets=self.test_targets,
            test_subject_ids=self.test_subject_ids,
        )
        if self.checkpoints and len(self.checkpoints) != len(self.config.training.final_seeds):
            raise ValueError("Loaded checkpoint count does not match ensemble seeds")


def torch_run_dir(
    config: TorchExperimentConfig,
    *,
    protocol: str,
    direction: str,
) -> Path:
    return (
        Path(config.artifacts.root)
        / config.model_id
        / build_torch_run_hash(config, protocol=protocol, direction=direction)
    )


def build_torch_direction_result(
    *,
    direction: EvaluationDirection,
    audit: ProtocolLeakageAudit,
    fitted: FittedTorchEnsemble,
    prediction: TorchEnsemblePrediction,
    targets: PixelTargetDataset,
    config: TorchExperimentConfig,
    training_seconds: float | None = None,
    prediction_seconds: float | None = None,
) -> TorchDirectionEvaluationResult:
    y_train = targets.y[fitted.training_target_indices]
    y_test = targets.y[prediction.test_target_indices]
    model_metrics = evaluate_prediction_matrix(
        y_test,
        prediction.predictions,
        prediction.scores,
    )
    model_bootstrap = bootstrap_subject_mean_balanced_accuracy(
        y_test,
        prediction.predictions,
        targets.subject_ids[prediction.test_target_indices],
        n_resamples=config.bootstrap_iterations,
        random_state=config.random_state,
    )
    baselines = tuple(
        TorchBaselineEvaluation(
            prediction=baseline,
            metrics=evaluate_prediction_matrix(
                y_test,
                baseline.predictions,
                baseline.probabilities,
            ),
        )
        for baseline in build_non_eeg_baselines(
            y_train,
            n_test_samples=y_test.shape[0],
            threshold=config.training.prediction_threshold,
            random_state=config.random_state,
        )
    )
    return TorchDirectionEvaluationResult(
        direction=direction,
        audit=audit,
        fitted=fitted,
        prediction=prediction,
        model_metrics=model_metrics,
        model_bootstrap=model_bootstrap,
        baselines=baselines,
        training_seconds=training_seconds,
        prediction_seconds=prediction_seconds,
    )


def write_torch_direction_run(
    result: TorchDirectionEvaluationResult,
    *,
    targets: PixelTargetDataset,
    config: TorchExperimentConfig,
    spectral_config_hash: str,
) -> Path:
    _validate_artifact_config(config.artifacts)
    _validate_direction_result(result, targets=targets, config=config)
    run_dir = torch_run_dir(
        config,
        protocol=result.direction.protocol,
        direction=result.direction.name,
    )
    root = run_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(f"Torch run already exists and is immutable: {run_dir}")
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}-", dir=root))
    try:
        _write_torch_payload(
            temporary_dir,
            result=result,
            targets=targets,
            config=config,
            spectral_config_hash=spectral_config_hash,
            config_hash=run_dir.name,
        )
        _publish_run(temporary_dir, run_dir=run_dir)
        return run_dir
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def load_torch_run(run_dir: Path, *, trusted: bool = False) -> LoadedTorchRun:
    resolved_dir = Path(run_dir)
    manifest_path = resolved_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Torch run manifest does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    _validate_manifest(resolved_dir, manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Torch artifact reader received a different schema version")

    config = TorchExperimentConfig.model_validate(_load_json(resolved_dir / "config.json"))
    evaluation = _load_json(resolved_dir / "evaluation.json")
    direction = evaluation.get("direction")
    if not isinstance(direction, dict):
        raise ValueError("Torch evaluation direction metadata is invalid")
    expected_hash = build_torch_run_hash(
        config,
        protocol=str(evaluation.get("protocol")),
        direction=str(direction.get("name")),
    )
    if manifest.get("config_hash") != expected_hash or resolved_dir.name != expected_hash:
        raise ValueError("Torch config hash does not match manifest and run directory")
    if manifest.get("model_id") != config.model_id:
        raise ValueError("Torch manifest model does not match configuration")

    environment = _load_json(resolved_dir / "environment.json")
    split = _load_json(resolved_dir / "split.json")
    preprocessing = _load_json(resolved_dir / "preprocessing.json")
    training = _load_json(resolved_dir / "training.json")
    arrays_dir = resolved_dir / "arrays"
    scores = _load_array(arrays_dir / "scores.npy", np.float64)
    predictions = _load_array(arrays_dir / "predictions.npy", np.int8)
    member_scores = _load_array(arrays_dir / "member_scores.npy", np.float64)
    test_balanced_accuracy = _load_array(arrays_dir / "test_balanced_accuracy.npy", np.float64)
    train_targets = _load_array(arrays_dir / "train_targets.npy", np.int8)
    test_targets = _load_array(arrays_dir / "test_targets.npy", np.int8)
    test_subject_ids = _load_array(arrays_dir / "test_subject_ids.npy", np.int64)
    normalization = _load_normalization(resolved_dir)
    baseline_scores, baseline_predictions = _load_baselines(
        resolved_dir,
        evaluation=evaluation,
    )
    _validate_evaluation_consistency(
        evaluation,
        split=split,
        training=training,
        scores=scores,
        predictions=predictions,
        member_scores=member_scores,
        test_balanced_accuracy=test_balanced_accuracy,
        test_targets=test_targets,
        test_subject_ids=test_subject_ids,
        baseline_scores=baseline_scores,
        baseline_predictions=baseline_predictions,
        config=config,
    )
    checkpoint_files = _checkpoint_files(training)
    checkpoint_paths = tuple(
        _resolve_checkpoint_path(
            resolved_dir,
            relative_path,
            manifest_files=set(manifest["files"]),
        )
        for relative_path in checkpoint_files
    )
    checkpoints: tuple[ModelCheckpointState, ...] = ()
    if trusted:
        checkpoints = tuple(
            _load_checkpoint(path, epoch=int(item["epoch"]))
            for path, item in zip(checkpoint_paths, training["ensemble_members"], strict=True)
        )
    return LoadedTorchRun(
        run_dir=resolved_dir,
        manifest=manifest,
        config=config,
        environment=environment,
        split=split,
        preprocessing=preprocessing,
        training=training,
        evaluation=evaluation,
        scores=scores,
        predictions=predictions,
        member_scores=member_scores,
        test_balanced_accuracy=test_balanced_accuracy,
        train_targets=train_targets,
        test_targets=test_targets,
        test_subject_ids=test_subject_ids,
        baseline_scores=baseline_scores,
        baseline_predictions=baseline_predictions,
        normalization=normalization,
        checkpoints=checkpoints,
    )


def replay_torch_predictions(
    run: LoadedTorchRun,
    *,
    source_dataset: CropSpectralDataset,
    targets: PixelTargetDataset,
    device: torch.device | str = "cpu",
    model_factory: ModelFactory = build_spectral_model,
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    if not run.checkpoints:
        raise PermissionError("Trusted replay requires `load_torch_run(..., trusted=True)`")
    spectral_hash = getattr(source_dataset, "config_hash", None)
    if spectral_hash is not None and spectral_hash != run.preprocessing["spectral_config_hash"]:
        raise ValueError("Spectral input configuration hash does not match persisted run")
    if source_dataset.method != run.config.training.method:
        raise ValueError("Spectral dataset method does not match persisted run")
    expected_keys = tuple(tuple(key) for key in run.split["test_sample_keys"])
    expected_rows = np.asarray(run.split["test_target_indices"], dtype=np.int64)
    if tuple(targets.sample_keys[int(row)] for row in expected_rows) != expected_keys:
        raise ValueError("Target dataset test sample keys do not match persisted run")

    from experiments.random_imagery_torch.training import predict_torch_ensemble

    # EnsembleMember normally requires final training history for freshly fitted models; persisted
    # replay only needs the trusted checkpoint and seed.
    rebuilt_members = []
    for item, checkpoint in zip(run.training["ensemble_members"], run.checkpoints, strict=True):
        member = object.__new__(EnsembleMember)
        object.__setattr__(member, "seed", int(item["seed"]))
        object.__setattr__(member, "history", ())
        object.__setattr__(member, "checkpoint", checkpoint)
        rebuilt_members.append(member)
    fitted = object.__new__(FittedTorchEnsemble)
    object.__setattr__(fitted, "architecture", run.config.training.architecture)
    object.__setattr__(fitted, "method", run.config.training.method)
    object.__setattr__(
        fitted,
        "input_shape",
        SpectralModelShape(
            input_planes=int(run.training["input_shape"]["input_planes"]),
            electrodes=int(run.training["input_shape"]["electrodes"]),
            width=int(run.training["input_shape"]["width"]),
        ),
    )
    object.__setattr__(
        fitted,
        "training_target_indices",
        np.asarray(run.split["train_target_indices"], dtype=np.int64),
    )
    object.__setattr__(
        fitted,
        "training_sample_keys",
        tuple(tuple(key) for key in run.split["train_sample_keys"]),
    )
    object.__setattr__(fitted, "normalization", run.normalization)
    object.__setattr__(
        fitted,
        "positive_weights",
        _load_array(run.run_dir / "arrays" / "positive_weights.npy", np.float32),
    )
    object.__setattr__(fitted, "selection", None)
    object.__setattr__(fitted, "members", tuple(rebuilt_members))
    object.__setattr__(fitted, "prediction_threshold", run.config.training.prediction_threshold)
    prediction = predict_torch_ensemble(
        fitted,
        source_dataset,
        targets,
        expected_rows,
        config=run.config.training,
        device=device,
        model_factory=model_factory,
    )
    return prediction.scores, prediction.predictions


def summarize_torch_runs(run_dirs: list[Path] | tuple[Path, ...]) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("At least one Torch run is required")
    runs = tuple(load_torch_run(path) for path in run_dirs)
    payload: dict[str, Any] = {
        "runs": [_torch_run_summary(run) for run in runs],
        "combined": None,
    }
    if len(runs) == 2 and {run.evaluation["protocol"] for run in runs} == {"within-subject"}:
        payload["combined"] = _combine_torch_runs(runs)
    elif len(runs) > 1 and {run.evaluation["protocol"] for run in runs} == {"within-subject"}:
        raise ValueError("Within-subject evaluation requires exactly two complementary directions")
    return payload


def compare_torch_runs(run_dirs: list[Path] | tuple[Path, ...]) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("Torch comparison requires at least two runs")
    runs = tuple(load_torch_run(path) for path in run_dirs)
    protocols = {(run.evaluation["protocol"], run.evaluation["direction"]["name"]) for run in runs}
    if len(protocols) != 1:
        raise ValueError("Compared Torch runs must use the same protocol and direction")
    reference_keys = tuple(tuple(key) for key in runs[0].split["test_sample_keys"])
    if any(tuple(tuple(key) for key in run.split["test_sample_keys"]) != reference_keys for run in runs[1:]):
        raise ValueError("Compared Torch runs must use identical ordered test sample keys")
    summaries = [
        {
            "run_dir": str(run.run_dir),
            "artifact_schema_version": SCHEMA_VERSION,
            "model_id": run.config.model_id,
            "score_semantics": "native_probability",
            "model_metrics": run.evaluation["model_metrics"],
            "model_bootstrap": run.evaluation["model_bootstrap"],
        }
        for run in runs
    ]
    reference_accuracy = summaries[0]["model_metrics"]["mean_balanced_accuracy"]
    for summary in summaries:
        summary["balanced_accuracy_difference_vs_first"] = (
            summary["model_metrics"]["mean_balanced_accuracy"] - reference_accuracy
        )
    return {
        "protocol": runs[0].evaluation["protocol"],
        "direction": runs[0].evaluation["direction"]["name"],
        "n_test_rows": len(reference_keys),
        "runs": summaries,
    }


def _write_torch_payload(
    run_dir: Path,
    *,
    result: TorchDirectionEvaluationResult,
    targets: PixelTargetDataset,
    config: TorchExperimentConfig,
    spectral_config_hash: str,
    config_hash: str,
) -> None:
    arrays_dir = run_dir / "arrays"
    checkpoints_dir = run_dir / "checkpoints"
    arrays_dir.mkdir()
    checkpoints_dir.mkdir()
    checkpoint_files = _write_checkpoints(checkpoints_dir, result.fitted)
    _write_json(run_dir / "config.json", config.model_dump(mode="json"))
    _write_json(run_dir / "environment.json", _build_environment_payload())
    _write_json(run_dir / "split.json", _build_split_payload(result, targets=targets))
    _write_json(
        run_dir / "preprocessing.json",
        {
            "spectral_config_hash": spectral_config_hash,
            "method": config.training.method,
            "input_config": config.spectral_input.model_dump(mode="json"),
            "preprocessing_overrides": config.preprocessing_overrides,
        },
    )
    _write_training_payload(run_dir / "training.json", result, checkpoint_files=checkpoint_files)
    _write_normalization(run_dir, result.fitted.normalization)
    _write_json(run_dir / "evaluation.json", _build_evaluation_payload(result))

    train_indices = result.fitted.training_target_indices
    test_indices = result.prediction.test_target_indices
    _write_array(arrays_dir / "scores.npy", result.prediction.scores)
    _write_array(arrays_dir / "predictions.npy", result.prediction.predictions)
    _write_array(arrays_dir / "member_scores.npy", result.prediction.member_scores)
    _write_array(arrays_dir / "test_balanced_accuracy.npy", result.model_metrics.per_pixel_balanced_accuracy)
    _write_array(arrays_dir / "train_targets.npy", targets.y[train_indices])
    _write_array(arrays_dir / "test_targets.npy", targets.y[test_indices])
    _write_array(arrays_dir / "test_subject_ids.npy", targets.subject_ids[test_indices])
    _write_array(arrays_dir / "positive_weights.npy", result.fitted.positive_weights)
    for baseline in result.baselines:
        name = baseline.prediction.name
        _write_array(arrays_dir / f"baseline_{name}_scores.npy", baseline.prediction.probabilities)
        _write_array(arrays_dir / f"baseline_{name}_predictions.npy", baseline.prediction.predictions)

    files = {
        path.relative_to(run_dir).as_posix(): {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "config_hash": config_hash,
            "model_id": config.model_id,
            "protocol": result.direction.protocol,
            "direction": result.direction.name,
            "generated_at": datetime.now(UTC).isoformat(),
            "writer": "experiments.random_imagery_torch.artifacts.write_torch_direction_run",
            "file_count": len(files),
            "files": files,
        },
    )
    _fsync_directory(run_dir)


def _write_checkpoints(
    checkpoints_dir: Path,
    fitted: FittedTorchEnsemble,
) -> list[str]:
    files = []
    for member in fitted.members:
        relative = f"checkpoints/seed_{member.seed}.pt"
        path = checkpoints_dir.parent / relative
        state = {name: tensor for name, tensor in member.checkpoint.state_dict.items()}
        with path.open("wb") as file:
            torch.save(state, file)
            file.flush()
            os.fsync(file.fileno())
        files.append(relative)
    return files


def _write_training_payload(
    path: Path,
    result: TorchDirectionEvaluationResult,
    *,
    checkpoint_files: list[str],
) -> None:
    fitted = result.fitted
    _write_json(
        path,
        {
            "architecture": fitted.architecture,
            "method": fitted.method,
            "input_shape": {
                "input_planes": fitted.input_shape.input_planes,
                "electrodes": fitted.input_shape.electrodes,
                "width": fitted.input_shape.width,
            },
            "parameter_count": _parameter_count_from_state(fitted.members[0].checkpoint.state_dict),
            "selected_epoch_count": fitted.selection.selected_epoch_count,
            "selection_seed": fitted.selection.selection_seed,
            "folds": [_fold_payload(fold_result) for fold_result in fitted.selection.folds],
            "ensemble_members": [
                {
                    "seed": member.seed,
                    "epoch": member.checkpoint.epoch,
                    "checkpoint_file": checkpoint_file,
                    "history": [
                        {"epoch": record.epoch, "train_bce": record.train_bce}
                        for record in member.history
                    ],
                }
                for member, checkpoint_file in zip(fitted.members, checkpoint_files, strict=True)
            ],
            "training_seconds": result.training_seconds,
            "prediction_seconds": result.prediction_seconds,
        },
    )


def _fold_payload(fold_result: Any) -> dict[str, Any]:
    return {
        "fold_index": fold_result.fold.fold_index,
        "train_target_indices": fold_result.fold.train_target_indices.tolist(),
        "validation_target_indices": fold_result.fold.validation_target_indices.tolist(),
        "train_subjects": list(fold_result.fold.train_subjects),
        "validation_subjects": list(fold_result.fold.validation_subjects),
        "best_epoch": fold_result.checkpoint.epoch,
        "stopped_epoch": fold_result.stopped_epoch,
        "normalization_fit_sample_keys": [list(key) for key in fold_result.normalization.fit_sample_keys],
        "positive_weights": fold_result.positive_weights.tolist(),
        "history": [
            {
                "epoch": record.epoch,
                "train_bce": record.train_bce,
                "validation_bce": record.validation_bce,
                "validation_balanced_accuracy": record.validation_balanced_accuracy,
            }
            for record in fold_result.history
        ],
    }


def _write_normalization(run_dir: Path, state: SpectralNormalizationState) -> None:
    arrays_dir = run_dir / "arrays"
    _write_array(arrays_dir / "normalization_frequencies.npy", state.frequencies)
    _write_array(arrays_dir / "normalization_mean.npy", state.mean)
    _write_array(arrays_dir / "normalization_scale.npy", state.scale)
    _write_array(arrays_dir / "normalization_zero_variance_mask.npy", state.zero_variance_mask)
    _write_json(
        run_dir / "normalization.json",
        {
            "method": state.method,
            "scaling": state.scaling,
            "eeg_channels": list(state.eeg_channels),
            "fit_sample_keys": [list(key) for key in state.fit_sample_keys],
            "observation_count": state.observation_count,
            "crop_bounds_seconds": list(state.crop_bounds_seconds),
            "log_epsilon": state.log_epsilon,
            "std_epsilon": state.std_epsilon,
            "arrays": {
                "frequencies": "arrays/normalization_frequencies.npy",
                "mean": "arrays/normalization_mean.npy",
                "scale": "arrays/normalization_scale.npy",
                "zero_variance_mask": "arrays/normalization_zero_variance_mask.npy",
            },
        },
    )


def _load_normalization(run_dir: Path) -> SpectralNormalizationState:
    payload = _load_json(run_dir / "normalization.json")
    arrays = payload.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("Normalization array metadata is invalid")
    return SpectralNormalizationState(
        method=payload["method"],
        scaling=payload["scaling"],
        frequencies=_load_array(_resolve_array_path(run_dir, arrays["frequencies"]), np.float32),
        eeg_channels=tuple(payload["eeg_channels"]),
        mean=_load_array(_resolve_array_path(run_dir, arrays["mean"]), np.float64),
        scale=_load_array(_resolve_array_path(run_dir, arrays["scale"]), np.float64),
        zero_variance_mask=_load_array(_resolve_array_path(run_dir, arrays["zero_variance_mask"]), np.bool_),
        fit_sample_keys=tuple(tuple(key) for key in payload["fit_sample_keys"]),
        observation_count=int(payload["observation_count"]),
        crop_bounds_seconds=tuple(payload["crop_bounds_seconds"]),
        log_epsilon=float(payload["log_epsilon"]),
        std_epsilon=float(payload["std_epsilon"]),
    )


def _build_split_payload(
    result: TorchDirectionEvaluationResult,
    *,
    targets: PixelTargetDataset,
) -> dict[str, Any]:
    direction = result.direction
    return {
        "protocol": direction.protocol,
        "direction": direction.name,
        "train_target_indices": direction.train_indices.tolist(),
        "test_target_indices": direction.test_indices.tolist(),
        "train_subjects": list(direction.train_subjects),
        "test_subjects": list(direction.test_subjects),
        "train_sample_keys": [list(key) for key in result.fitted.training_sample_keys],
        "test_sample_keys": [list(key) for key in result.prediction.test_sample_keys],
        "test_image_fingerprints": [
            targets.image_fingerprints[int(index)]
            for index in direction.test_indices
        ],
    }


def _build_evaluation_payload(result: TorchDirectionEvaluationResult) -> dict[str, Any]:
    direction = result.direction
    audit = result.audit
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": f"{result.fitted.architecture}-{result.fitted.method}-multilabel",
        "score_semantics": "native_probability",
        "protocol": direction.protocol,
        "direction": {
            "name": direction.name,
            "label": direction.label,
            "train_trial": direction.train_trial,
            "test_trial": direction.test_trial,
        },
        "eligible_subjects": list(direction.eligible_subjects),
        "excluded_subjects": list(direction.excluded_subjects),
        "split": {
            "n_train_rows": int(direction.train_indices.size),
            "n_test_rows": int(direction.test_indices.size),
            "train_subjects": list(direction.train_subjects),
            "test_subjects": list(direction.test_subjects),
        },
        "split_audit": {
            "overlapping_subjects": list(audit.overlapping_subjects),
            "overlapping_sample_keys": [list(key) for key in audit.overlapping_sample_keys],
            "overlapping_seeds": list(audit.overlapping_seeds),
            "overlapping_image_fingerprints": list(audit.overlapping_image_fingerprints),
            "overlapping_trial_numbers": list(audit.overlapping_trial_numbers),
            "train_positive_counts": audit.train_positive_counts.tolist(),
            "test_positive_counts": audit.test_positive_counts.tolist(),
            "all_tasks_have_both_classes": audit.all_tasks_have_both_classes,
            "subject_contract_satisfied": audit.subject_contract_satisfied,
            "trial_contract_satisfied": audit.trial_contract_satisfied,
            "has_forbidden_leakage": audit.has_forbidden_leakage,
        },
        "prediction_threshold": result.prediction.threshold,
        "member_seeds": list(result.prediction.member_seeds),
        "model_metrics": _metrics_payload(result.model_metrics),
        "model_bootstrap": _bootstrap_payload(result.model_bootstrap),
        "baselines": [
            {
                "name": baseline.prediction.name,
                "metrics": _metrics_payload(baseline.metrics),
                "array_files": {
                    "scores": f"arrays/baseline_{baseline.prediction.name}_scores.npy",
                    "predictions": f"arrays/baseline_{baseline.prediction.name}_predictions.npy",
                },
            }
            for baseline in result.baselines
        ],
    }


def _validate_evaluation_consistency(
    evaluation: dict[str, Any],
    *,
    split: dict[str, Any],
    training: dict[str, Any],
    scores: NDArray[np.float64],
    predictions: NDArray[np.int8],
    member_scores: NDArray[np.float64],
    test_balanced_accuracy: NDArray[np.float64],
    test_targets: NDArray[np.int8],
    test_subject_ids: NDArray[np.int64],
    baseline_scores: dict[str, NDArray[np.float64]],
    baseline_predictions: dict[str, NDArray[np.int8]],
    config: TorchExperimentConfig,
) -> None:
    if evaluation.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Torch evaluation payload declares the wrong schema")
    if evaluation.get("model_id") != config.model_id:
        raise ValueError("Torch evaluation model does not match configuration")
    direction = evaluation.get("direction")
    if not isinstance(direction, dict) or direction.get("name") != split.get("direction"):
        raise ValueError("Torch evaluation direction does not match split metadata")
    if evaluation.get("protocol") != split.get("protocol"):
        raise ValueError("Torch evaluation protocol does not match split metadata")
    audit = evaluation.get("split_audit")
    if not isinstance(audit, dict) or audit.get("has_forbidden_leakage") is not False:
        raise ValueError("Torch split audit is invalid or reports forbidden leakage")
    if not np.array_equal(scores, member_scores.mean(axis=0, dtype=np.float64)):
        raise ValueError("Stored Torch scores are not the mean member probabilities")
    if not np.array_equal(predictions, (scores >= float(evaluation["prediction_threshold"])).astype(np.int8)):
        raise ValueError("Stored Torch predictions differ from thresholded scores")
    if evaluation.get("member_seeds") != [member["seed"] for member in training["ensemble_members"]]:
        raise ValueError("Torch member seeds differ between training and evaluation metadata")

    metrics = evaluate_prediction_matrix(test_targets, predictions, scores)
    _assert_metrics_payload(evaluation.get("model_metrics"), expected=metrics, label="model")
    if not np.allclose(test_balanced_accuracy, metrics.per_pixel_balanced_accuracy, rtol=0.0, atol=1e-12):
        raise ValueError("Stored Torch balanced accuracy differs from arrays")
    bootstrap = bootstrap_subject_mean_balanced_accuracy(
        test_targets,
        predictions,
        test_subject_ids,
        n_resamples=config.bootstrap_iterations,
        random_state=config.random_state,
    )
    if evaluation.get("model_bootstrap") != _bootstrap_payload(bootstrap):
        raise ValueError("Stored Torch bootstrap differs from arrays")
    for item in evaluation["baselines"]:
        name = item["name"]
        _assert_metrics_payload(
            item.get("metrics"),
            expected=evaluate_prediction_matrix(test_targets, baseline_predictions[name], baseline_scores[name]),
            label=f"baseline {name}",
        )


def _validate_direction_result(
    result: TorchDirectionEvaluationResult,
    *,
    targets: PixelTargetDataset,
    config: TorchExperimentConfig,
) -> None:
    if f"{result.fitted.architecture}-{result.fitted.method}-multilabel" != config.model_id:
        raise ValueError("Fitted Torch model does not match artifact configuration")
    expected_train_keys = tuple(targets.sample_keys[int(index)] for index in result.direction.train_indices)
    expected_test_keys = tuple(targets.sample_keys[int(index)] for index in result.direction.test_indices)
    if result.fitted.training_sample_keys != expected_train_keys:
        raise ValueError("Fitted Torch training sample keys do not match targets")
    if result.prediction.test_sample_keys != expected_test_keys:
        raise ValueError("Torch prediction sample keys do not match targets")


def _validate_loaded_arrays(
    *,
    scores: NDArray[np.float64],
    predictions: NDArray[np.int8],
    member_scores: NDArray[np.float64],
    test_balanced_accuracy: NDArray[np.float64],
    train_targets: NDArray[np.int8],
    test_targets: NDArray[np.int8],
    test_subject_ids: NDArray[np.int64],
) -> None:
    if scores.ndim != 2 or scores.dtype != np.dtype(np.float64):
        raise TypeError("Loaded Torch scores must be a float64 matrix")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Loaded Torch scores must be finite probabilities")
    n_test, n_targets = scores.shape
    if predictions.shape != scores.shape or predictions.dtype != np.dtype(np.int8):
        raise TypeError("Loaded Torch predictions must be an int8 matrix matching scores")
    if member_scores.shape != (3, n_test, n_targets) or member_scores.dtype != np.dtype(np.float64):
        raise TypeError("Loaded Torch member scores must have shape (3, sample, target)")
    if train_targets.ndim != 2 or train_targets.dtype != np.dtype(np.int8):
        raise TypeError("Loaded Torch train targets must be a binary int8 matrix")
    if test_targets.shape != (n_test, n_targets) or test_targets.dtype != np.dtype(np.int8):
        raise TypeError("Loaded Torch test targets must match scores")
    if test_subject_ids.shape != (n_test,) or test_subject_ids.dtype != np.dtype(np.int64):
        raise TypeError("Loaded Torch test subject IDs must match test rows")
    if test_balanced_accuracy.shape != (n_targets,) or test_balanced_accuracy.dtype != np.dtype(np.float64):
        raise TypeError("Loaded Torch balanced accuracy must match target columns")


def _load_baselines(
    run_dir: Path,
    *,
    evaluation: dict[str, Any],
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.int8]]]:
    scores: dict[str, NDArray[np.float64]] = {}
    predictions: dict[str, NDArray[np.int8]] = {}
    for item in evaluation["baselines"]:
        name = item["name"]
        files = item["array_files"]
        scores[name] = _load_array(_resolve_array_path(run_dir, files["scores"]), np.float64)
        predictions[name] = _load_array(_resolve_array_path(run_dir, files["predictions"]), np.int8)
    return scores, predictions


def _checkpoint_files(training: dict[str, Any]) -> tuple[str, ...]:
    members = training.get("ensemble_members")
    if not isinstance(members, list) or len(members) != 3:
        raise ValueError("Torch training metadata requires three ensemble members")
    files = tuple(member.get("checkpoint_file") for member in members)
    if any(not isinstance(path, str) for path in files):
        raise ValueError("Torch checkpoint file metadata is invalid")
    return files  # type: ignore[return-value]


def _resolve_array_path(run_dir: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("Torch array filename must be a string")
    path = Path(relative_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "arrays"
        or path.suffix != ".npy"
    ):
        raise ValueError(f"Unsafe Torch array filename: {relative_path!r}")
    return run_dir / path


def _resolve_checkpoint_path(
    run_dir: Path,
    relative_path: object,
    *,
    manifest_files: set[str],
) -> Path:
    if not isinstance(relative_path, str):
        raise ValueError("Torch checkpoint filename must be a string")
    path = Path(relative_path)
    normalized = path.as_posix()
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "checkpoints"
        or path.suffix != ".pt"
    ):
        raise ValueError(f"Unsafe Torch checkpoint filename: {relative_path!r}")
    if normalized not in manifest_files:
        raise ValueError(f"Torch checkpoint is not present in manifest: {relative_path}")
    return run_dir / path


def _load_checkpoint(path: Path, *, epoch: int) -> ModelCheckpointState:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Torch checkpoint is not a state dict: {path}")
    return ModelCheckpointState(epoch=epoch, state_dict=state)


def _metrics_payload(metrics: PredictionMetrics) -> dict[str, Any]:
    return {
        "per_pixel_balanced_accuracy": metrics.per_pixel_balanced_accuracy.tolist(),
        "per_pixel_macro_f1": metrics.per_pixel_macro_f1.tolist(),
        "per_pixel_score_mse": metrics.per_pixel_score_mse.tolist(),
        "per_sample_iou": metrics.per_sample_iou.tolist(),
        "mean_balanced_accuracy": metrics.mean_balanced_accuracy,
        "mean_macro_f1": metrics.mean_macro_f1,
        "mean_score_mse": metrics.mean_score_mse,
        "mean_sample_iou": metrics.mean_sample_iou,
        "micro_iou": metrics.micro_iou,
        "bit_accuracy": metrics.bit_accuracy,
        "exact_match_accuracy": metrics.exact_match_accuracy,
        "mean_hamming_distance": metrics.mean_hamming_distance,
        "hamming_loss": metrics.hamming_loss,
    }


def _bootstrap_payload(interval: SubjectBootstrapInterval) -> dict[str, Any]:
    return {
        "metric": "mean_balanced_accuracy",
        "unit": "subject",
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "confidence_level": interval.confidence_level,
        "n_resamples": interval.n_resamples,
        "n_attempts": interval.n_attempts,
    }


def _assert_metrics_payload(payload: object, *, expected: PredictionMetrics, label: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Stored Torch {label} metrics are invalid")
    expected_payload = _metrics_payload(expected)
    for key, expected_value in expected_payload.items():
        actual_value = payload.get(key)
        if isinstance(expected_value, list):
            if not isinstance(actual_value, list) or not np.allclose(
                actual_value,
                expected_value,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"Stored Torch {label} metric differs from arrays: {key}")
        elif actual_value != expected_value:
            raise ValueError(f"Stored Torch {label} metric differs from arrays: {key}")


def _torch_run_summary(run: LoadedTorchRun) -> dict[str, Any]:
    return {
        "run_dir": str(run.run_dir),
        "artifact_schema_version": SCHEMA_VERSION,
        **run.evaluation,
    }


def _combine_torch_runs(runs: tuple[LoadedTorchRun, ...]) -> dict[str, Any]:
    by_direction = {run.evaluation["direction"]["name"]: run for run in runs}
    expected = ("trial-1-to-trial-2", "trial-2-to-trial-1")
    if set(by_direction) != set(expected):
        raise ValueError("Two-run aggregation requires complementary within-subject directions")
    ordered = tuple(by_direction[name] for name in expected)
    if ordered[0].config != ordered[1].config:
        raise ValueError("Combined Torch runs must use identical configurations")
    first_keys = {tuple(key) for key in ordered[0].split["test_sample_keys"]}
    second_keys = {tuple(key) for key in ordered[1].split["test_sample_keys"]}
    if first_keys & second_keys:
        raise ValueError("Combined Torch directions overlap test sample keys")
    test_targets = np.concatenate([run.test_targets for run in ordered], axis=0)
    scores = np.concatenate([run.scores for run in ordered], axis=0)
    predictions = np.concatenate([run.predictions for run in ordered], axis=0)
    subject_ids = np.concatenate([run.test_subject_ids for run in ordered])
    metrics = evaluate_prediction_matrix(test_targets, predictions, scores)
    bootstrap = bootstrap_subject_mean_balanced_accuracy(
        test_targets,
        predictions,
        subject_ids,
        n_resamples=ordered[0].config.bootstrap_iterations,
        random_state=ordered[0].config.random_state,
    )
    return {
        "model_id": ordered[0].config.model_id,
        "protocol": "within-subject",
        "direction_names": list(expected),
        "run_dirs": [str(run.run_dir) for run in ordered],
        "n_test_rows": int(test_targets.shape[0]),
        "n_subjects": int(np.unique(subject_ids).size),
        "model_metrics": _metrics_payload(metrics),
        "model_bootstrap": _bootstrap_payload(bootstrap),
    }


def _parameter_count_from_state(state: Any) -> int:
    return int(sum(tensor.numel() for tensor in state.values() if isinstance(tensor, torch.Tensor)))


def _validate_artifact_config(config: ArtifactConfig) -> None:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError("Torch artifact workflow requires schema_version=1")
    if config.overwrite:
        raise ValueError("Torch artifact workflow does not permit destructive overwrite")


def _validate_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("file_count") != len(files):
        raise ValueError("Torch manifest file inventory is invalid")
    expected_files = set(files) | {"manifest.json"}
    actual_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(f"Torch file inventory mismatch; missing={missing}, unexpected={unexpected}")
    for relative_path, metadata in files.items():
        path = run_dir / relative_path
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid Torch manifest metadata for {relative_path}")
        if path.stat().st_size != metadata.get("size"):
            raise ValueError(f"Torch file size mismatch: {relative_path}")
        if _sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Torch file hash mismatch: {relative_path}")


def _build_environment_payload() -> dict[str, Any]:
    package_names = ("mne", "numpy", "omegaconf", "pydantic", "scikit-learn", "scipy", "torch")
    git_commit, git_dirty = _git_state()
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            package_name: importlib.metadata.version(package_name)
            for package_name in package_names
        },
        "torch_cuda_available": cuda_available,
        "torch_cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def _publish_run(temporary_dir: Path, *, run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(f"Torch run already exists and is immutable: {run_dir}")
    os.replace(temporary_dir, run_dir)
    _fsync_directory(run_dir.parent)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid Torch artifact JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Torch artifact JSON file must contain an object: {path}")
    return payload


def _write_array(path: Path, array: NDArray[Any]) -> None:
    with path.open("wb") as file:
        np.save(file, array, allow_pickle=False)
        file.flush()
        os.fsync(file.fileno())


def _load_array(path: Path, dtype: np.dtype[Any] | type[Any]) -> NDArray[Any]:
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"Torch artifact array {path.name} has unexpected dtype {array.dtype}")
    array.setflags(write=False)
    return array


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "LoadedTorchRun",
    "TorchBaselineEvaluation",
    "TorchDirectionEvaluationResult",
    "build_torch_direction_result",
    "compare_torch_runs",
    "load_torch_run",
    "replay_torch_predictions",
    "summarize_torch_runs",
    "torch_run_dir",
    "write_torch_direction_run",
]
