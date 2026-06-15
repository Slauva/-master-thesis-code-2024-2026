import random
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from experiments.random_imagery.data import audit_evaluation_direction
from experiments.random_imagery.schemas import EvaluationDirection, PixelTargetDataset
from experiments.random_imagery_torch.config import TorchTrainingConfig
from experiments.random_imagery_torch.models import (
    ArchitectureName,
    SpectralModel,
    SpectralModelShape,
    build_spectral_model,
)
from experiments.random_imagery_torch.normalization import fit_spectral_normalization
from experiments.random_imagery_torch.schemas import TorchSpectralInputBatch
from experiments.random_imagery_torch.spectral_dataset import CropSpectralDataset
from experiments.random_imagery_torch.torch_dataset import (
    TorchSpectralInputDataset,
    collate_spectral_inputs,
)
from experiments.random_imagery_torch.training_schemas import (
    EnsembleMember,
    EpochSelectionResult,
    FinalEpochRecord,
    FittedTorchEnsemble,
    FoldSelectionResult,
    GroupedTrainingFold,
    ModelCheckpointState,
    TorchEnsemblePrediction,
    ValidationEpochRecord,
)

ModelFactory = Callable[..., SpectralModel]


@dataclass(frozen=True, slots=True)
class _ValidationMetrics:
    bce: float
    balanced_accuracy: float


def build_grouped_training_folds(
    targets: PixelTargetDataset,
    direction: EvaluationDirection,
    *,
    config: TorchTrainingConfig,
) -> tuple[GroupedTrainingFold, GroupedTrainingFold, GroupedTrainingFold]:
    train_rows = direction.train_indices
    groups = targets.subject_ids[train_rows]
    if np.unique(groups).size < config.validation_folds:
        raise ValueError("Grouped validation requires at least one subject per fold")

    splitter = GroupKFold(
        n_splits=config.validation_folds,
        shuffle=True,
        random_state=config.selection_seed,
    )
    folds: list[GroupedTrainingFold] = []
    validation_rows: list[NDArray[np.int64]] = []
    for fold_index, (local_train, local_validation) in enumerate(
        splitter.split(np.zeros((train_rows.size, 1)), groups=groups)
    ):
        fold_train = np.sort(train_rows[local_train].astype(np.int64, copy=False))
        fold_validation = np.sort(
            train_rows[local_validation].astype(np.int64, copy=False)
        )
        _require_class_complete(targets.y[fold_train], label=f"fold {fold_index} train")
        _require_class_complete(
            targets.y[fold_validation],
            label=f"fold {fold_index} validation",
        )
        validation_rows.append(fold_validation)
        folds.append(
            GroupedTrainingFold(
                fold_index=fold_index,
                train_target_indices=fold_train,
                validation_target_indices=fold_validation,
                train_subjects=tuple(
                    int(value)
                    for value in np.unique(targets.subject_ids[fold_train])
                ),
                validation_subjects=tuple(
                    int(value)
                    for value in np.unique(targets.subject_ids[fold_validation])
                ),
            )
        )
    if not np.array_equal(
        np.sort(np.concatenate(validation_rows)),
        direction.train_indices,
    ):
        raise ValueError("Grouped validation folds must partition the direction training rows")
    if len(folds) != 3:
        raise RuntimeError("Torch epoch selection requires exactly three folds")
    return folds[0], folds[1], folds[2]


def compute_positive_class_weights(y: ArrayLike) -> NDArray[np.float32]:
    targets = np.asarray(y)
    if targets.ndim != 2 or targets.shape[1] != 36 or not np.isin(targets, (0, 1)).all():
        raise ValueError("Positive weights require a binary (sample, 36) target matrix")
    _require_class_complete(targets, label="positive-weight training")
    positives = targets.sum(axis=0, dtype=np.float64)
    negatives = targets.shape[0] - positives
    weights = np.asarray(negatives / positives, dtype=np.float32)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Positive weights must be finite and positive")
    weights.setflags(write=False)
    return weights


def fit_torch_ensemble(
    source_dataset: CropSpectralDataset,
    targets: PixelTargetDataset,
    direction: EvaluationDirection,
    *,
    config: TorchTrainingConfig,
    device: torch.device | str | None = None,
    model_factory: ModelFactory = build_spectral_model,
) -> FittedTorchEnsemble:
    _validate_fit_inputs(source_dataset, targets, direction, config=config)
    resolved_device = resolve_training_device(
        config.device if device is None else device
    )
    folds = build_grouped_training_folds(targets, direction, config=config)
    fold_results: list[FoldSelectionResult] = []
    expected_shape: SpectralModelShape | None = None
    for fold in folds:
        train_keys = _sample_keys(targets, fold.train_target_indices)
        normalization = fit_spectral_normalization(source_dataset, train_keys)
        positive_weights = compute_positive_class_weights(
            targets.y[fold.train_target_indices]
        )
        training_dataset = TorchSpectralInputDataset(
            source_dataset,
            targets,
            fold.train_target_indices,
            normalization,
        )
        validation_dataset = TorchSpectralInputDataset(
            source_dataset,
            targets,
            fold.validation_target_indices,
            normalization,
        )
        input_shape = _infer_input_shape(training_dataset)
        if expected_shape is None:
            expected_shape = input_shape
        elif input_shape != expected_shape:
            raise ValueError("Grouped folds produced inconsistent model input shapes")
        fold_results.append(
            _fit_validation_fold(
                training_dataset,
                validation_dataset,
                fold=fold,
                normalization=normalization,
                positive_weights=positive_weights,
                input_shape=input_shape,
                config=config,
                device=resolved_device,
                model_factory=model_factory,
            )
        )

    selection = EpochSelectionResult(
        folds=(fold_results[0], fold_results[1], fold_results[2]),
        selected_epoch_count=int(
            np.median([result.checkpoint.epoch for result in fold_results])
        ),
        selection_seed=config.selection_seed,
    )
    training_rows = direction.train_indices.copy()
    training_keys = _sample_keys(targets, training_rows)
    final_normalization = fit_spectral_normalization(source_dataset, training_keys)
    final_positive_weights = compute_positive_class_weights(targets.y[training_rows])
    final_dataset = TorchSpectralInputDataset(
        source_dataset,
        targets,
        training_rows,
        final_normalization,
    )
    final_shape = _infer_input_shape(final_dataset)
    if expected_shape is not None and final_shape != expected_shape:
        raise ValueError("Final training input shape differs from grouped validation")
    members = tuple(
        _fit_fixed_epochs(
            final_dataset,
            positive_weights=final_positive_weights,
            input_shape=final_shape,
            seed=seed,
            epochs=selection.selected_epoch_count,
            config=config,
            device=resolved_device,
            model_factory=model_factory,
        )
        for seed in config.final_seeds
    )
    return FittedTorchEnsemble(
        architecture=config.architecture,
        method=config.method,
        input_shape=final_shape,
        training_target_indices=training_rows,
        training_sample_keys=training_keys,
        normalization=final_normalization,
        positive_weights=final_positive_weights.copy(),
        selection=selection,
        members=(members[0], members[1], members[2]),
        prediction_threshold=config.prediction_threshold,
    )


def predict_torch_ensemble(
    fitted: FittedTorchEnsemble,
    source_dataset: CropSpectralDataset,
    targets: PixelTargetDataset,
    test_target_indices: NDArray[np.int64],
    *,
    config: TorchTrainingConfig,
    device: torch.device | str | None = None,
    model_factory: ModelFactory = build_spectral_model,
) -> TorchEnsemblePrediction:
    rows = np.asarray(test_target_indices)
    if rows.ndim != 1 or rows.dtype != np.dtype(np.int64) or rows.size < 1:
        raise TypeError("Test target indices must be a non-empty int64 vector")
    if not np.array_equal(rows, np.unique(rows)):
        raise ValueError("Test target indices must be sorted and unique")
    if np.intersect1d(rows, fitted.training_target_indices).size:
        raise ValueError("Prediction rows must be disjoint from fitted training rows")
    if config.architecture != fitted.architecture or config.method != fitted.method:
        raise ValueError("Prediction configuration does not match the fitted ensemble")
    if config.final_seeds != tuple(member.seed for member in fitted.members):
        raise ValueError("Prediction ensemble seeds do not match the fitted ensemble")
    if config.prediction_threshold != fitted.prediction_threshold:
        raise ValueError("Prediction threshold does not match the fitted ensemble")
    if source_dataset.method != fitted.method:
        raise ValueError("Prediction spectral method does not match the fitted ensemble")
    if fitted.normalization.fit_sample_keys != fitted.training_sample_keys:
        raise ValueError("Fitted normalization provenance is inconsistent")

    resolved_device = resolve_training_device(
        config.device if device is None else device
    )
    test_dataset = TorchSpectralInputDataset(
        source_dataset,
        targets,
        rows,
        fitted.normalization,
    )
    member_scores = np.stack(
        [
            _predict_member(
                member,
                test_dataset,
                architecture=fitted.architecture,
                input_shape=fitted.input_shape,
                config=config,
                device=resolved_device,
                model_factory=model_factory,
            )
            for member in fitted.members
        ]
    ).astype(np.float64, copy=False)
    scores = member_scores.mean(axis=0, dtype=np.float64)
    predictions = (scores >= fitted.prediction_threshold).astype(np.int8)
    return TorchEnsemblePrediction(
        test_target_indices=rows.copy(),
        test_sample_keys=_sample_keys(targets, rows),
        member_seeds=tuple(member.seed for member in fitted.members),
        member_scores=member_scores,
        scores=scores,
        predictions=predictions,
        threshold=fitted.prediction_threshold,
    )


def resolve_training_device(
    requested: str | torch.device,
) -> torch.device:
    if str(requested) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Torch spectral training supports only CPU or CUDA devices")
    return device


def _fit_validation_fold(
    training_dataset: TorchSpectralInputDataset,
    validation_dataset: TorchSpectralInputDataset,
    *,
    fold: GroupedTrainingFold,
    normalization: object,
    positive_weights: NDArray[np.float32],
    input_shape: SpectralModelShape,
    config: TorchTrainingConfig,
    device: torch.device,
    model_factory: ModelFactory,
) -> FoldSelectionResult:
    seed = config.selection_seed
    with _deterministic_torch(seed, enabled=config.deterministic):
        model = model_factory(
            config.architecture,
            input_shape=input_shape,
            dropout_rate=config.dropout_rate,
        ).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.from_numpy(positive_weights.copy()).to(device)
        )
        training_loader = _build_loader(
            training_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            seed=seed,
            device=device,
        )
        validation_loader = _build_loader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            seed=seed,
            device=device,
        )
        history: list[ValidationEpochRecord] = []
        best_checkpoint: ModelCheckpointState | None = None
        best_accuracy = -np.inf
        best_bce = np.inf
        patience_accuracy = -np.inf
        patience_bce = np.inf
        epochs_without_improvement = 0
        for epoch in range(1, config.maximum_epochs + 1):
            train_bce = _train_epoch(
                model,
                training_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            validation = _evaluate_epoch(
                model,
                validation_loader,
                criterion=criterion,
                device=device,
                threshold=config.prediction_threshold,
            )
            record = ValidationEpochRecord(
                epoch=epoch,
                train_bce=train_bce,
                validation_bce=validation.bce,
                validation_balanced_accuracy=validation.balanced_accuracy,
            )
            history.append(record)
            if (
                record.validation_balanced_accuracy > best_accuracy
                or (
                    record.validation_balanced_accuracy == best_accuracy
                    and record.validation_bce < best_bce
                )
            ):
                best_accuracy = record.validation_balanced_accuracy
                best_bce = record.validation_bce
                best_checkpoint = _snapshot_checkpoint(model, epoch=epoch)

            materially_improved = (
                record.validation_balanced_accuracy
                > patience_accuracy + config.early_stopping_min_delta
                or (
                    abs(record.validation_balanced_accuracy - patience_accuracy)
                    <= config.early_stopping_min_delta
                    and record.validation_bce
                    < patience_bce - config.early_stopping_min_delta
                )
            )
            if materially_improved:
                patience_accuracy = record.validation_balanced_accuracy
                patience_bce = record.validation_bce
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break
    if best_checkpoint is None:
        raise RuntimeError("Validation training did not produce a checkpoint")
    return FoldSelectionResult(
        fold=fold,
        normalization=normalization,  # type: ignore[arg-type]
        positive_weights=positive_weights.copy(),
        history=tuple(history),
        checkpoint=best_checkpoint,
        stopped_epoch=history[-1].epoch,
    )


def _fit_fixed_epochs(
    training_dataset: TorchSpectralInputDataset,
    *,
    positive_weights: NDArray[np.float32],
    input_shape: SpectralModelShape,
    seed: int,
    epochs: int,
    config: TorchTrainingConfig,
    device: torch.device,
    model_factory: ModelFactory,
) -> EnsembleMember:
    with _deterministic_torch(seed, enabled=config.deterministic):
        model = model_factory(
            config.architecture,
            input_shape=input_shape,
            dropout_rate=config.dropout_rate,
        ).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.from_numpy(positive_weights.copy()).to(device)
        )
        loader = _build_loader(
            training_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            seed=seed,
            device=device,
        )
        history = tuple(
            FinalEpochRecord(
                epoch=epoch,
                train_bce=_train_epoch(
                    model,
                    loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    gradient_clip_norm=config.gradient_clip_norm,
                ),
            )
            for epoch in range(1, epochs + 1)
        )
        checkpoint = _snapshot_checkpoint(model, epoch=epochs)
    return EnsembleMember(seed=seed, history=history, checkpoint=checkpoint)


def _train_epoch(
    model: SpectralModel,
    loader: DataLoader[TorchSpectralInputBatch],
    *,
    optimizer: AdamW,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_rows = 0
    for batch in loader:
        moved = batch.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits = model(moved.model_inputs)
        loss = criterion(logits, moved.targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became non-finite")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not gradients or any(
            gradient is None or not torch.isfinite(gradient).all()
            for gradient in gradients
        ):
            raise FloatingPointError("Training gradients became missing or non-finite")
        norm = clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not torch.isfinite(torch.as_tensor(norm)):
            raise FloatingPointError("Gradient norm became non-finite")
        optimizer.step()
        model.project_max_norm_()
        batch_rows = moved.targets.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_rows
        n_rows += batch_rows
    if n_rows < 1:
        raise RuntimeError("Training loader produced no rows")
    return total_loss / n_rows


@torch.no_grad()
def _evaluate_epoch(
    model: SpectralModel,
    loader: DataLoader[TorchSpectralInputBatch],
    *,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    threshold: float,
) -> _ValidationMetrics:
    model.eval()
    total_loss = 0.0
    targets: list[NDArray[np.float32]] = []
    predictions: list[NDArray[np.int8]] = []
    n_rows = 0
    for batch in loader:
        moved = batch.to(device, non_blocking=device.type == "cuda")
        logits = model(moved.model_inputs)
        loss = criterion(logits, moved.targets)
        probabilities = torch.sigmoid(logits)
        if not torch.isfinite(loss) or not torch.isfinite(probabilities).all():
            raise FloatingPointError("Validation outputs became non-finite")
        batch_rows = moved.targets.shape[0]
        total_loss += float(loss.cpu()) * batch_rows
        n_rows += batch_rows
        targets.append(moved.targets.cpu().numpy())
        predictions.append(
            (probabilities >= threshold).to(torch.int8).cpu().numpy()
        )
    if n_rows < 1:
        raise RuntimeError("Validation loader produced no rows")
    return _ValidationMetrics(
        bce=total_loss / n_rows,
        balanced_accuracy=_mean_per_pixel_balanced_accuracy(
            np.concatenate(targets, axis=0),
            np.concatenate(predictions, axis=0),
        ),
    )


@torch.no_grad()
def _predict_member(
    member: EnsembleMember,
    dataset: TorchSpectralInputDataset,
    *,
    architecture: ArchitectureName,
    input_shape: SpectralModelShape,
    config: TorchTrainingConfig,
    device: torch.device,
    model_factory: ModelFactory,
) -> NDArray[np.float64]:
    model = model_factory(
        architecture,
        input_shape=input_shape,
        dropout_rate=config.dropout_rate,
    )
    model.load_state_dict(member.checkpoint.state_dict, strict=True)
    model.to(device)
    model.eval()
    loader = _build_loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=member.seed,
        device=device,
    )
    scores: list[NDArray[np.float64]] = []
    observed_rows: list[NDArray[np.int64]] = []
    for batch in loader:
        moved = batch.to(device, non_blocking=device.type == "cuda")
        probabilities = torch.sigmoid(model(moved.model_inputs))
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("Ensemble prediction produced non-finite scores")
        scores.append(probabilities.cpu().numpy().astype(np.float64))
        observed_rows.append(batch.target_row_indices.numpy())
    if not scores:
        raise RuntimeError("Prediction loader produced no rows")
    if not np.array_equal(
        np.concatenate(observed_rows),
        dataset.target_row_indices,
    ):
        raise ValueError("Prediction loader changed target row order")
    return np.concatenate(scores, axis=0)


def _build_loader(
    dataset: TorchSpectralInputDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader[TorchSpectralInputBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_spectral_inputs,
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
    )


def _infer_input_shape(dataset: TorchSpectralInputDataset) -> SpectralModelShape:
    model_input = dataset[0].model_input
    if model_input.ndim != 3:
        raise ValueError("Torch spectral samples must have three model-input dimensions")
    return SpectralModelShape(
        input_planes=int(model_input.shape[0]),
        electrodes=int(model_input.shape[1]),
        width=int(model_input.shape[2]),
    )


def _snapshot_checkpoint(
    model: SpectralModel,
    *,
    epoch: int,
) -> ModelCheckpointState:
    return ModelCheckpointState(
        epoch=epoch,
        state_dict={
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
    )


def _mean_per_pixel_balanced_accuracy(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.int8],
) -> float:
    targets = np.asarray(y_true)
    labels = np.asarray(y_pred)
    if targets.shape != labels.shape or targets.ndim != 2:
        raise ValueError("Balanced accuracy requires matching target and prediction matrices")
    _require_class_complete(targets, label="balanced-accuracy validation")
    positives = targets == 1
    negatives = ~positives
    sensitivity = np.sum(labels == 1, axis=0, where=positives) / positives.sum(axis=0)
    specificity = np.sum(labels == 0, axis=0, where=negatives) / negatives.sum(axis=0)
    score = float(np.mean((sensitivity + specificity) / 2.0))
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Balanced accuracy must be finite and in [0, 1]")
    return score


def _require_class_complete(y: ArrayLike, *, label: str) -> None:
    targets = np.asarray(y)
    if targets.ndim != 2 or targets.shape[0] < 2 or not np.isin(targets, (0, 1)).all():
        raise ValueError(f"{label} targets must be a non-empty binary matrix")
    positives = targets.sum(axis=0)
    if np.any((positives == 0) | (positives == targets.shape[0])):
        raise ValueError(f"{label} lacks both classes in at least one pixel")


def _sample_keys(
    targets: PixelTargetDataset,
    rows: Sequence[int] | NDArray[np.int64],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(targets.sample_keys[int(row)] for row in rows)


def _validate_fit_inputs(
    source_dataset: CropSpectralDataset,
    targets: PixelTargetDataset,
    direction: EvaluationDirection,
    *,
    config: TorchTrainingConfig,
) -> None:
    if source_dataset.method != config.method:
        raise ValueError("Training configuration method does not match the spectral dataset")
    audit = audit_evaluation_direction(targets, direction)
    if audit.has_forbidden_leakage:
        raise ValueError("Torch training direction violates the evaluation leakage contract")
    if not audit.all_tasks_have_both_classes:
        raise ValueError("Torch training direction lacks both classes in a pixel task")
    _require_class_complete(targets.y[direction.train_indices], label="outer training")


@contextmanager
def _deterministic_torch(seed: int, *, enabled: bool) -> Iterator[None]:
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
