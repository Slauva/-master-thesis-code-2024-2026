from pathlib import Path

import numpy as np
import pytest

from experiments.logistic_regression import (
    AlignedTrainingFeatures,
    FeatureFamily,
    PixelTargetDataset,
    SubjectSplit,
    build_aligned_training_features,
    build_grouped_pixel_cross_validation,
    load_logistic_regression_config,
    screen_feature_families,
)
from features import FeatureBlock, FeatureSet
from utils.datasets import RandomSample


class RecordingFeatureDataset:
    def __init__(self, feature_sets: dict[tuple[int, int, int], FeatureSet]) -> None:
        self.feature_sets = feature_sets
        self.requested_keys: list[tuple[int, int, int]] = []

    def __len__(self) -> int:
        return len(self.feature_sets)

    def __getitem__(self, key: int | tuple[int, int, int]) -> FeatureSet:
        if isinstance(key, int):
            raise AssertionError("Alignment must use canonical sample keys")
        self.requested_keys.append(key)
        return self.feature_sets[key]


def _sample(subject: int) -> RandomSample:
    return RandomSample(
        subject_id=subject,
        trial_number=1,
        Exec_Block_Index=1,
        eeg_path=Path(f"S_{subject}/patt_EEG_1.fif"),
        eog_path=Path(f"S_{subject}/patt_EOG_1.fif"),
        img=np.zeros((6, 6), dtype=np.int8).tolist(),
        seed=subject,
    )


def _feature_set(subject: int) -> FeatureSet:
    return FeatureSet(
        sample=_sample(subject),
        blocks=(
            FeatureBlock(
                name="time",
                layout="channel_features",
                values=np.asarray([[[subject, subject + 0.5]]], dtype=np.float32),
                feature_names=("mean", "variance"),
            ),
            FeatureBlock(
                name="spectral",
                layout="channel_features",
                values=np.asarray([[[subject + 1.0]]], dtype=np.float32),
                feature_names=("alpha",),
            ),
        ),
        window_bounds_seconds=np.asarray([[0.5, 15.5]], dtype=np.float64),
        eeg_channels=("Fz",),
        analysis_sfreq=125.0,
    )


def _targets(n_subjects: int) -> PixelTargetDataset:
    sample_keys = tuple((subject, 1, 1) for subject in range(1, n_subjects + 1))
    subject_ids = np.arange(1, n_subjects + 1, dtype=np.int64)
    y = np.asarray(
        [[subject % 2, (subject // 2) % 2] for subject in range(1, n_subjects + 1)],
        dtype=np.int8,
    )
    return PixelTargetDataset(
        y=y,
        pixel_names=("pixel_r0_c0", "pixel_r0_c1"),
        sample_keys=sample_keys,
        subject_ids=subject_ids,
        trial_numbers=np.ones(n_subjects, dtype=np.int64),
        block_indices=np.ones(n_subjects, dtype=np.int64),
        seeds=subject_ids.copy(),
        image_fingerprints=tuple(f"fingerprint-{subject}" for subject in subject_ids),
    )


def test_alignment_loads_only_outer_training_keys() -> None:
    targets = _targets(8)
    split = SubjectSplit(
        train_indices=np.arange(6, dtype=np.int64),
        test_indices=np.arange(6, 8, dtype=np.int64),
        train_subjects=tuple(range(1, 7)),
        test_subjects=(7, 8),
        n_samples=8,
        random_state=42,
        test_size=0.25,
    )
    dataset = RecordingFeatureDataset(
        {key: _feature_set(key[0]) for key in targets.sample_keys}
    )

    aligned = build_aligned_training_features(
        dataset,
        targets=targets,
        split=split,
        candidates=(("time",), ("time", "spectral")),
    )

    assert dataset.requested_keys == list(targets.sample_keys[:6])
    assert all(key not in dataset.requested_keys for key in targets.sample_keys[6:])
    assert tuple(family.X.shape for family in aligned.families) == ((6, 2), (6, 3))
    assert aligned.sample_keys == targets.sample_keys[:6]
    np.testing.assert_array_equal(aligned.target_row_indices, split.train_indices)


def test_grouped_pixel_cv_is_deterministic_disjoint_and_class_complete() -> None:
    config = load_logistic_regression_config(
        overrides={"cross_validation": {"n_splits": 3}}
    )
    groups = np.repeat(np.arange(1, 13, dtype=np.int64), 2)
    y = np.column_stack(
        (
            np.tile([0, 1], 12),
            np.asarray([(subject + row) % 2 for subject in range(12) for row in range(2)]),
        )
    ).astype(np.int8)

    first = build_grouped_pixel_cross_validation(y, groups=groups, config=config.cross_validation)
    second = build_grouped_pixel_cross_validation(y, groups=groups, config=config.cross_validation)

    for first_fold, second_fold in zip(first.folds, second.folds, strict=True):
        np.testing.assert_array_equal(first_fold.train_indices, second_fold.train_indices)
        np.testing.assert_array_equal(first_fold.validation_indices, second_fold.validation_indices)
        assert not set(first_fold.train_subjects) & set(first_fold.validation_subjects)
        pixel = first_fold.pixel_index
        assert np.unique(y[first_fold.train_indices, pixel]).size == 2
        assert np.unique(y[first_fold.validation_indices, pixel]).size == 2


def test_screening_fits_transforms_only_on_fold_training_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_logistic_regression_config(
        overrides={
            "cross_validation": {"n_splits": 3},
            "feature_screening": {
                "select_k": 2,
                "candidates": [["time"]],
            },
        }
    )
    groups = np.repeat(np.arange(1, 13, dtype=np.int64), 2)
    y = np.column_stack(
        (
            np.tile([0, 1], 12),
            np.asarray([(subject + row) % 2 for subject in range(12) for row in range(2)]),
        )
    ).astype(np.int8)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(y.shape[0], 4)).astype(np.float64)
    features = AlignedTrainingFeatures(
        families=(
            FeatureFamily(
                block_names=("time",),
                X=X,
                feature_names=("a", "b", "c", "d"),
            ),
        ),
        target_row_indices=np.arange(y.shape[0], dtype=np.int64),
        sample_keys=tuple((index + 1, 1, 1) for index in range(y.shape[0])),
        subject_ids=groups,
        window_bounds_seconds=np.tile([0.5, 15.5], (y.shape[0], 1)).astype(np.float64),
    )
    cross_validation = build_grouped_pixel_cross_validation(
        y,
        groups=groups,
        config=config.cross_validation,
    )

    from experiments.logistic_regression import screening as screening_module

    original_fit = screening_module.StandardScaler.fit
    fitted_row_counts: list[int] = []

    def recording_fit(self: object, values: np.ndarray, target: object = None) -> object:
        fitted_row_counts.append(values.shape[0])
        return original_fit(self, values, target)

    monkeypatch.setattr(screening_module.StandardScaler, "fit", recording_fit)
    result = screen_feature_families(
        features,
        y=y,
        cross_validation=cross_validation,
        config=config.feature_screening,
        random_state=config.random_state,
    )

    expected_train_sizes = sorted(
        fold.train_indices.size
        for pixel_index in range(y.shape[1])
        for fold in cross_validation.for_pixel(pixel_index)
    )
    assert sorted(fitted_row_counts) == expected_train_sizes
    assert all(row_count < y.shape[0] for row_count in fitted_row_counts)
    assert result.selected_block_names == ("time",)


def test_screening_uses_candidate_order_to_break_exact_ties() -> None:
    config = load_logistic_regression_config(
        overrides={
            "cross_validation": {"n_splits": 3},
            "feature_screening": {
                "select_k": 2,
                "candidates": [["time"], ["spectral"]],
            },
        }
    )
    groups = np.repeat(np.arange(1, 13, dtype=np.int64), 2)
    y = np.tile([0, 1], 12).reshape(-1, 1).astype(np.int8)
    X = np.column_stack((y[:, 0], 1 - y[:, 0])).astype(np.float64)
    shared = {
        "X": X,
        "feature_names": ("feature_0", "feature_1"),
    }
    features = AlignedTrainingFeatures(
        families=(
            FeatureFamily(block_names=("time",), **shared),
            FeatureFamily(block_names=("spectral",), **shared),
        ),
        target_row_indices=np.arange(y.shape[0], dtype=np.int64),
        sample_keys=tuple((index + 1, 1, 1) for index in range(y.shape[0])),
        subject_ids=groups,
        window_bounds_seconds=np.tile([0.5, 15.5], (y.shape[0], 1)).astype(np.float64),
    )
    cross_validation = build_grouped_pixel_cross_validation(
        y,
        groups=groups,
        config=config.cross_validation,
    )

    result = screen_feature_families(
        features,
        y=y,
        cross_validation=cross_validation,
        config=config.feature_screening,
        random_state=config.random_state,
    )

    assert result.candidates[0].mean_score == result.candidates[1].mean_score
    assert result.selected_block_names == ("time",)
