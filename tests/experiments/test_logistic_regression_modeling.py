from pathlib import Path

import numpy as np
import pytest

from experiments.logistic_regression import (
    PixelTargetDataset,
    SubjectSplit,
    load_logistic_regression_config,
    run_per_pixel_grid_search,
)
from features import FeatureBlock, FeatureSet
from utils.datasets import RandomSample


class EventFeatureDataset:
    def __init__(
        self,
        feature_sets: dict[tuple[int, int, int], FeatureSet],
        events: list[str],
    ) -> None:
        self.feature_sets = feature_sets
        self.events = events

    def __len__(self) -> int:
        return len(self.feature_sets)

    def __getitem__(self, key: int | tuple[int, int, int]) -> FeatureSet:
        if isinstance(key, int):
            raise AssertionError("Modeling must use canonical sample keys")
        self.events.append(f"load:{key[0]}")
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


def _feature_set(subject: int, targets: np.ndarray) -> FeatureSet:
    rng = np.random.default_rng(subject)
    values = rng.normal(scale=0.05, size=8)
    values[0] += 2.0 * targets[0]
    values[1] += 2.0 * targets[1]
    return FeatureSet(
        sample=_sample(subject),
        blocks=(
            FeatureBlock(
                name="lbp",
                layout="channel_histogram",
                values=values.reshape(1, 1, -1).astype(np.float32),
                feature_names=tuple(f"code_{index:03d}" for index in range(values.size)),
            ),
        ),
        window_bounds_seconds=np.asarray([[0.5, 15.5]], dtype=np.float64),
        eeg_channels=("Fz",),
        analysis_sfreq=125.0,
    )


def _targets(n_subjects: int = 18) -> PixelTargetDataset:
    subject_ids = np.arange(1, n_subjects + 1, dtype=np.int64)
    y = np.column_stack(
        (
            subject_ids % 2,
            (subject_ids // 2) % 2,
        )
    ).astype(np.int8)
    return PixelTargetDataset(
        y=y,
        pixel_names=("pixel_r0_c0", "pixel_r0_c1"),
        sample_keys=tuple((int(subject), 1, 1) for subject in subject_ids),
        subject_ids=subject_ids,
        trial_numbers=np.ones(n_subjects, dtype=np.int64),
        block_indices=np.ones(n_subjects, dtype=np.int64),
        seeds=subject_ids.copy(),
        image_fingerprints=tuple(f"fingerprint-{subject}" for subject in subject_ids),
    )


def _split() -> SubjectSplit:
    return SubjectSplit(
        train_indices=np.arange(15, dtype=np.int64),
        test_indices=np.arange(15, 18, dtype=np.int64),
        train_subjects=tuple(range(1, 16)),
        test_subjects=(16, 17, 18),
        n_samples=18,
        random_state=42,
        test_size=1 / 6,
    )


def _config() -> object:
    return load_logistic_regression_config(
        overrides={
            "cross_validation": {"n_splits": 3},
            "grid_search": {
                "select_k": [2, 4],
                "c_values": [0.1, 1.0],
                "penalties": ["l1", "l2"],
                "class_weights": [None, "balanced"],
                "max_iter": 1000,
                "n_jobs": 1,
            },
        }
    )


def test_grid_search_delays_test_loading_and_returns_complete_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets()
    split = _split()
    events: list[str] = []
    dataset = EventFeatureDataset(
        {
            key: _feature_set(key[0], targets.y[index])
            for index, key in enumerate(targets.sample_keys)
        },
        events,
    )
    config = _config()

    from experiments.logistic_regression import modeling as modeling_module

    original_fit = modeling_module.GridSearchCV.fit
    original_scaler_fit = modeling_module.StandardScaler.fit
    scaler_row_counts: list[int] = []

    def recording_search_fit(self: object, values: np.ndarray, target: np.ndarray) -> object:
        events.append("grid-fit")
        return original_fit(self, values, target)

    def recording_scaler_fit(self: object, values: np.ndarray, target: object = None) -> object:
        scaler_row_counts.append(values.shape[0])
        return original_scaler_fit(self, values, target)

    monkeypatch.setattr(modeling_module.GridSearchCV, "fit", recording_search_fit)
    monkeypatch.setattr(modeling_module.StandardScaler, "fit", recording_scaler_fit)
    result = run_per_pixel_grid_search(
        dataset,
        targets=targets,
        split=split,
        block_names=("lbp",),
        cross_validation_config=config.cross_validation,
        grid_search_config=config.grid_search,
        scoring=config.cross_validation.scoring,
        threshold=config.prediction_threshold,
        random_state=config.random_state,
    )

    test_load_positions = [
        index
        for index, event in enumerate(events)
        if event in {"load:16", "load:17", "load:18"}
    ]
    grid_fit_positions = [index for index, event in enumerate(events) if event == "grid-fit"]
    assert len(grid_fit_positions) == 2
    assert min(test_load_positions) > max(grid_fit_positions)
    assert result.probabilities.shape == (3, 2)
    assert result.predictions.shape == (3, 2)
    assert result.test_balanced_accuracy.shape == (2,)
    assert len(result.fitted_models.models) == 2
    assert all(model.pipeline.memory is None for model in result.fitted_models.models)
    assert all(len(model.candidate_scores) == 16 for model in result.fitted_models.models)
    assert all(model.selected_feature_indices.size in (2, 4) for model in result.fitted_models.models)
    assert all(model.n_iter >= 0 for model in result.fitted_models.models)
    assert all(row_count <= split.train_indices.size for row_count in scaler_row_counts)
    assert split.test_indices.size not in scaler_row_counts


def test_grid_search_is_deterministic() -> None:
    targets = _targets()
    split = _split()
    config = _config()

    def run() -> object:
        dataset = EventFeatureDataset(
            {
                key: _feature_set(key[0], targets.y[index])
                for index, key in enumerate(targets.sample_keys)
            },
            [],
        )
        return run_per_pixel_grid_search(
            dataset,
            targets=targets,
            split=split,
            block_names=("lbp",),
            cross_validation_config=config.cross_validation,
            grid_search_config=config.grid_search,
            scoring=config.cross_validation.scoring,
            threshold=config.prediction_threshold,
            random_state=config.random_state,
        )

    first = run()
    second = run()

    np.testing.assert_array_equal(first.predictions, second.predictions)
    np.testing.assert_allclose(first.probabilities, second.probabilities, rtol=0.0, atol=0.0)
    assert tuple(
        model.best_hyperparameters for model in first.fitted_models.models
    ) == tuple(model.best_hyperparameters for model in second.fitted_models.models)
    assert tuple(model.best_cv_score for model in first.fitted_models.models) == tuple(
        model.best_cv_score for model in second.fitted_models.models
    )
