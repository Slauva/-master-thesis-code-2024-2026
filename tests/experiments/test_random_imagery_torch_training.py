import hashlib
from dataclasses import replace
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from experiments.random_imagery import build_random_imagery_targets
from experiments.random_imagery.schemas import EvaluationDirection, PixelTargetDataset
from experiments.random_imagery_torch import (
    CropSpectralSample,
    SpectralInputConfig,
    SpectralModel,
    SpectralModelShape,
    TorchTrainingConfig,
    build_grouped_training_folds,
    compute_positive_class_weights,
    fit_torch_ensemble,
    predict_torch_ensemble,
)
from preprocessors import load_preprocessing_config
from utils.datasets import RandomSample


class TinySpectralModel(SpectralModel):
    architecture = "eegnet"

    def __init__(self, *, input_shape: SpectralModelShape, n_outputs: int = 36) -> None:
        super().__init__(input_shape=input_shape, n_outputs=n_outputs)
        self.head = nn.Linear(prod(input_shape.tensor_shape), n_outputs)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_input(inputs)
        return self.head(inputs.flatten(1))


def _tiny_model_factory(
    architecture: str,
    *,
    input_shape: SpectralModelShape,
    dropout_rate: float | None = None,
) -> SpectralModel:
    del architecture, dropout_rate
    return TinySpectralModel(input_shape=input_shape)


class TrackingCropDataset:
    def __init__(self, samples: list[CropSpectralSample]) -> None:
        self._samples = tuple(samples)
        self._by_key = {sample.sample_key: sample for sample in samples}
        self.samples = tuple(sample.sample for sample in samples)
        self.input_config = SpectralInputConfig()
        self.method = "fft"
        self.preprocessing_config = load_preprocessing_config("fft")
        self.requested_keys: list[tuple[int, int, int]] = []

    def __getitem__(self, key: int | tuple[int, int, int]) -> CropSpectralSample:
        if isinstance(key, int):
            sample = self._samples[key]
        else:
            sample = self._by_key[key]
        self.requested_keys.append(sample.sample_key)
        return sample


def _synthetic_inputs() -> tuple[TrackingCropDataset, PixelTargetDataset, EvaluationDirection]:
    samples: list[CropSpectralSample] = []
    rng = np.random.default_rng(123)
    for subject in range(1, 9):
        base = rng.integers(0, 2, size=36, dtype=np.int8)
        while base.sum() in {0, 36}:
            base = rng.integers(0, 2, size=36, dtype=np.int8)
        for block, image in enumerate((base, 1 - base), start=1):
            samples.append(_crop_sample(subject=subject, block=block, image=image))
    dataset = TrackingCropDataset(samples)
    targets = build_random_imagery_targets(dataset.samples)
    train_indices = np.arange(12, dtype=np.int64)
    test_indices = np.arange(12, 16, dtype=np.int64)
    return (
        dataset,
        targets,
        EvaluationDirection(
            protocol="cross-subject",
            name="cross-subject",
            label="cross-subject",
            train_indices=train_indices,
            test_indices=test_indices,
            train_subjects=(1, 2, 3, 4, 5, 6),
            test_subjects=(7, 8),
            eligible_subjects=(1, 2, 3, 4, 5, 6, 7, 8),
            excluded_subjects=(),
            n_samples=16,
        ),
    )


def _crop_sample(
    *,
    subject: int,
    block: int,
    image: np.ndarray[Any, np.dtype[np.int8]],
) -> CropSpectralSample:
    frequencies = np.arange(2.0, 41.0, dtype=np.float32)
    channel_offsets = np.arange(4, dtype=np.float32)[:, np.newaxis]
    frequency_offsets = frequencies[np.newaxis, :] / 100.0
    power = 1.0 + channel_offsets + frequency_offsets + subject / 10.0 + block / 100.0
    sample = RandomSample(
        subject_id=subject,
        trial_number=1,
        Exec_Block_Index=block,
        eeg_path=Path(f"subject-{subject}-block-{block}-eeg.fif"),
        eog_path=Path(f"subject-{subject}-block-{block}-eog.fif"),
        img=image.reshape(6, 6).tolist(),
        seed=10_000 + subject * 10 + block,
    )
    return CropSpectralSample(
        sample=sample,
        eeg_power=power.astype(np.float32),
        frequencies=frequencies,
        times=None,
        eeg_channels=("Fz", "Cz", "Pz", "Oz"),
        source_sfreq=1_000.0,
        analysis_sfreq=125.0,
        method="fft",
        scaling="psd",
        crop_bounds_seconds=(0.5, 15.5),
    )


def _config(**overrides: Any) -> TorchTrainingConfig:
    values = {
        "architecture": "eegnet",
        "method": "fft",
        "batch_size": 4,
        "maximum_epochs": 2,
        "early_stopping_patience": 2,
        "selection_seed": 7,
        "final_seeds": (11, 12, 13),
        "device": "cpu",
    }
    values.update(overrides)
    return TorchTrainingConfig(**values)


def test_grouped_training_folds_are_subject_disjoint_and_class_complete() -> None:
    _, targets, direction = _synthetic_inputs()
    folds = build_grouped_training_folds(targets, direction, config=_config())

    assert tuple(fold.fold_index for fold in folds) == (0, 1, 2)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([fold.validation_target_indices for fold in folds])),
        direction.train_indices,
    )
    for fold in folds:
        assert not (set(fold.train_subjects) & set(fold.validation_subjects))
        for rows in (fold.train_target_indices, fold.validation_target_indices):
            positives = targets.y[rows].sum(axis=0)
            assert np.all((positives > 0) & (positives < rows.size))


def test_positive_class_weights_are_train_only_and_reject_missing_classes() -> None:
    _, targets, direction = _synthetic_inputs()

    weights = compute_positive_class_weights(targets.y[direction.train_indices])

    np.testing.assert_allclose(weights, np.ones(36, dtype=np.float32))
    broken = targets.y[direction.train_indices].copy()
    broken[:, 0] = 0
    with pytest.raises(ValueError, match="lacks both classes"):
        compute_positive_class_weights(broken)


def test_fit_torch_ensemble_uses_no_outer_test_samples_and_records_fold_provenance() -> None:
    dataset, targets, direction = _synthetic_inputs()
    config = _config()

    fitted = fit_torch_ensemble(
        dataset,  # type: ignore[arg-type]
        targets,
        direction,
        config=config,
        model_factory=_tiny_model_factory,
    )

    train_keys = {targets.sample_keys[int(row)] for row in direction.train_indices}
    test_keys = {targets.sample_keys[int(row)] for row in direction.test_indices}
    assert set(dataset.requested_keys) <= train_keys
    assert not (set(dataset.requested_keys) & test_keys)
    assert fitted.training_sample_keys == tuple(
        targets.sample_keys[int(row)] for row in direction.train_indices
    )
    assert fitted.normalization.fit_sample_keys == fitted.training_sample_keys
    assert tuple(member.seed for member in fitted.members) == config.final_seeds
    assert fitted.selection.selected_epoch_count == int(
        np.median([fold.checkpoint.epoch for fold in fitted.selection.folds])
    )
    for fold_result in fitted.selection.folds:
        fold_train_keys = tuple(
            targets.sample_keys[int(row)]
            for row in fold_result.fold.train_target_indices
        )
        fold_validation_keys = {
            targets.sample_keys[int(row)]
            for row in fold_result.fold.validation_target_indices
        }
        assert fold_result.normalization.fit_sample_keys == fold_train_keys
        assert not (set(fold_train_keys) & fold_validation_keys)
        np.testing.assert_allclose(
            fold_result.positive_weights,
            np.ones(36, dtype=np.float32),
        )


def test_predict_torch_ensemble_materializes_test_after_fit_and_returns_mean_scores() -> None:
    dataset, targets, direction = _synthetic_inputs()
    config = _config()
    fitted = fit_torch_ensemble(
        dataset,  # type: ignore[arg-type]
        targets,
        direction,
        config=config,
        model_factory=_tiny_model_factory,
    )
    requested_after_fit = list(dataset.requested_keys)

    prediction = predict_torch_ensemble(
        fitted,
        dataset,  # type: ignore[arg-type]
        targets,
        direction.test_indices,
        config=config,
        model_factory=_tiny_model_factory,
    )

    assert requested_after_fit == dataset.requested_keys[: len(requested_after_fit)]
    assert {targets.sample_keys[int(row)] for row in direction.test_indices} <= set(
        dataset.requested_keys[len(requested_after_fit) :]
    )
    assert prediction.member_scores.shape == (3, 4, 36)
    assert prediction.scores.shape == (4, 36)
    np.testing.assert_allclose(prediction.scores, prediction.member_scores.mean(axis=0))
    np.testing.assert_array_equal(
        prediction.predictions,
        (prediction.scores >= config.prediction_threshold).astype(np.int8),
    )
    model_prediction = prediction.to_model_prediction()
    assert model_prediction.diagnostics.score_semantics == "native_probability"


def test_training_and_prediction_are_deterministic_for_fixed_seeds() -> None:
    config = _config(maximum_epochs=1, early_stopping_patience=1)
    first_dataset, first_targets, first_direction = _synthetic_inputs()
    second_dataset, second_targets, second_direction = _synthetic_inputs()

    first = predict_torch_ensemble(
        fit_torch_ensemble(
            first_dataset,  # type: ignore[arg-type]
            first_targets,
            first_direction,
            config=config,
            model_factory=_tiny_model_factory,
        ),
        first_dataset,  # type: ignore[arg-type]
        first_targets,
        first_direction.test_indices,
        config=config,
        model_factory=_tiny_model_factory,
    )
    second = predict_torch_ensemble(
        fit_torch_ensemble(
            second_dataset,  # type: ignore[arg-type]
            second_targets,
            second_direction,
            config=config,
            model_factory=_tiny_model_factory,
        ),
        second_dataset,  # type: ignore[arg-type]
        second_targets,
        second_direction.test_indices,
        config=config,
        model_factory=_tiny_model_factory,
    )

    np.testing.assert_allclose(first.member_scores, second.member_scores, rtol=0.0, atol=0.0)
    assert _checkpoint_digest(first_dataset.requested_keys) == _checkpoint_digest(
        second_dataset.requested_keys
    )


def test_fit_rejects_outer_training_pixels_without_both_classes() -> None:
    dataset, targets, direction = _synthetic_inputs()
    broken_y = targets.y.copy()
    broken_y[direction.train_indices, 0] = 0
    broken_targets = replace(targets, y=broken_y)

    with pytest.raises(ValueError, match="lacks both classes"):
        fit_torch_ensemble(
            dataset,  # type: ignore[arg-type]
            broken_targets,
            direction,
            config=_config(),
            model_factory=_tiny_model_factory,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_training_smoke() -> None:
    dataset, targets, direction = _synthetic_inputs()

    fitted = fit_torch_ensemble(
        dataset,  # type: ignore[arg-type]
        targets,
        direction,
        config=_config(device="cuda", maximum_epochs=1, early_stopping_patience=1),
        model_factory=_tiny_model_factory,
    )

    assert fitted.members[0].checkpoint.epoch == 1


def _checkpoint_digest(keys: list[tuple[int, int, int]]) -> str:
    return hashlib.sha256(repr(keys).encode("utf-8")).hexdigest()
