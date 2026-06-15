from pathlib import Path

import numpy as np
import pytest

from experiments.logistic_regression import (
    PixelTargetDataset,
    load_logistic_regression_config,
    run_evaluation_protocol,
)
from features import FeatureBlock, FeatureSet
from utils.datasets import RandomSample


class SyntheticProtocolDataset:
    def __init__(self, feature_sets: dict[tuple[int, int, int], FeatureSet]) -> None:
        self.feature_sets = feature_sets

    def __len__(self) -> int:
        return len(self.feature_sets)

    def __getitem__(self, key: int | tuple[int, int, int]) -> FeatureSet:
        if isinstance(key, int):
            raise AssertionError("Protocol runner must use canonical sample keys")
        return self.feature_sets[key]


def _protocol_inputs(
    *,
    n_subjects: int = 12,
) -> tuple[SyntheticProtocolDataset, PixelTargetDataset]:
    sample_keys = tuple(
        (subject, trial, block)
        for subject in range(1, n_subjects + 1)
        for trial in (1, 2)
        for block in (1, 2, 3)
    )
    subject_ids = np.asarray([key[0] for key in sample_keys], dtype=np.int64)
    trial_numbers = np.asarray([key[1] for key in sample_keys], dtype=np.int64)
    block_indices = np.asarray([key[2] for key in sample_keys], dtype=np.int64)
    y = np.asarray(
        [
            [(subject + block) % 2, ((subject // 2) + block) % 2]
            for subject, _, block in sample_keys
        ],
        dtype=np.int8,
    )
    targets = PixelTargetDataset(
        y=y,
        pixel_names=("pixel_r0_c0", "pixel_r0_c1"),
        sample_keys=sample_keys,
        subject_ids=subject_ids,
        trial_numbers=trial_numbers,
        block_indices=block_indices,
        seeds=np.arange(1, len(sample_keys) + 1, dtype=np.int64),
        image_fingerprints=tuple(f"fingerprint-{index}" for index in range(len(sample_keys))),
    )

    feature_sets: dict[tuple[int, int, int], FeatureSet] = {}
    for row_index, key in enumerate(sample_keys):
        subject, trial, block = key
        rng = np.random.default_rng(subject * 100 + trial * 10 + block)
        time_values = rng.normal(scale=0.05, size=4)
        spectral_values = rng.normal(scale=0.05, size=4)
        time_values[:2] += 2.0 * y[row_index]
        spectral_values[:2] += 1.5 * y[row_index]
        sample = RandomSample(
            subject_id=subject,
            trial_number=trial,
            Exec_Block_Index=block,
            eeg_path=Path(f"S_{subject}/Trial_{trial}/patt_EEG_{block}.fif"),
            eog_path=Path(f"S_{subject}/Trial_{trial}/patt_EOG_{block}.fif"),
            img=np.zeros((6, 6), dtype=np.int8).tolist(),
            seed=row_index + 1,
        )
        feature_sets[key] = FeatureSet(
            sample=sample,
            blocks=(
                FeatureBlock(
                    name="time",
                    layout="channel_features",
                    values=time_values.reshape(1, 1, -1).astype(np.float32),
                    feature_names=tuple(f"time_{index}" for index in range(4)),
                ),
                FeatureBlock(
                    name="spectral",
                    layout="channel_features",
                    values=spectral_values.reshape(1, 1, -1).astype(np.float32),
                    feature_names=tuple(f"spectral_{index}" for index in range(4)),
                ),
            ),
            window_bounds_seconds=np.asarray([[0.5, 15.5]], dtype=np.float64),
            eeg_channels=("Fz",),
            analysis_sfreq=125.0,
        )
    return SyntheticProtocolDataset(feature_sets), targets


def _runner_config() -> object:
    return load_logistic_regression_config(
        overrides={
            "split": {"test_size": 0.25},
            "cross_validation": {"n_splits": 3},
            "feature_screening": {
                "select_k": 2,
                "max_iter": 1000,
                "candidates": [["time"], ["spectral"]],
            },
            "grid_search": {
                "select_k": [2],
                "c_values": [1.0],
                "penalties": ["l2"],
                "class_weights": ["balanced"],
                "max_iter": 1000,
                "n_jobs": 1,
            },
            "bootstrap_iterations": 50,
        }
    )


def test_within_subject_runner_orders_train_decisions_before_test_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, targets = _protocol_inputs()
    config = _runner_config()
    events: list[str] = []

    from experiments.logistic_regression import runner as runner_module

    original_training = runner_module.build_aligned_training_features
    original_screen = runner_module.screen_feature_families
    original_fit = runner_module.fit_pixel_models
    original_test = runner_module.build_aligned_feature_partition
    original_predict = runner_module.predict_pixel_models
    original_combine = runner_module._combine_within_subject_results

    def record_training(*args: object, **kwargs: object) -> object:
        result = original_training(*args, **kwargs)
        events.append(f"screen-data:{result.sample_keys[0][1]}")
        return result

    def record_screen(*args: object, **kwargs: object) -> object:
        features = args[0]
        events.append(f"screen:{features.sample_keys[0][1]}")
        return original_screen(*args, **kwargs)

    def record_fit(*args: object, **kwargs: object) -> object:
        features = args[0]
        events.append(f"fit:{features.sample_keys[0][1]}")
        return original_fit(*args, **kwargs)

    def record_test(*args: object, **kwargs: object) -> object:
        row_indices = kwargs["row_indices"]
        trial = int(targets.trial_numbers[int(row_indices[0])])
        events.append(f"test-data:{trial}")
        return original_test(*args, **kwargs)

    def record_predict(*args: object, **kwargs: object) -> object:
        features = kwargs["test_features"]
        events.append(f"predict:{features.sample_keys[0][1]}")
        return original_predict(*args, **kwargs)

    def record_combine(*args: object, **kwargs: object) -> object:
        events.append("combine")
        return original_combine(*args, **kwargs)

    monkeypatch.setattr(runner_module, "build_aligned_training_features", record_training)
    monkeypatch.setattr(runner_module, "screen_feature_families", record_screen)
    monkeypatch.setattr(runner_module, "fit_pixel_models", record_fit)
    monkeypatch.setattr(runner_module, "build_aligned_feature_partition", record_test)
    monkeypatch.setattr(runner_module, "predict_pixel_models", record_predict)
    monkeypatch.setattr(runner_module, "_combine_within_subject_results", record_combine)

    result = run_evaluation_protocol(
        "within-subject",
        config=config,
        dataset=dataset,
        targets=targets,
    )

    assert events == [
        "screen-data:1",
        "screen:1",
        "fit:1",
        "test-data:2",
        "predict:2",
        "screen-data:2",
        "screen:2",
        "fit:2",
        "test-data:1",
        "predict:1",
        "combine",
    ]
    assert result.combined is not None
    assert tuple(result.combined.direction_names) == (
        "trial-1-to-trial-2",
        "trial-2-to-trial-1",
    )
    assert result.combined.y_true.shape == (72, 2)
    np.testing.assert_array_equal(
        np.unique(result.combined.subject_ids, return_counts=True)[1],
        np.full(12, 6),
    )
    assert set(result.combined.target_indices.tolist()) == set(range(72))
    assert all(not direction.audit.has_forbidden_leakage for direction in result.directions)


def test_protocol_runner_is_deterministic_for_both_protocols() -> None:
    config = _runner_config()

    def run(protocol: str) -> object:
        dataset, targets = _protocol_inputs()
        return run_evaluation_protocol(
            protocol,  # type: ignore[arg-type]
            config=config,
            dataset=dataset,
            targets=targets,
        )

    first_cross = run("cross-subject")
    second_cross = run("cross-subject")
    np.testing.assert_array_equal(
        first_cross.directions[0].grid_search.predictions,
        second_cross.directions[0].grid_search.predictions,
    )
    np.testing.assert_allclose(
        first_cross.directions[0].grid_search.probabilities,
        second_cross.directions[0].grid_search.probabilities,
        rtol=0.0,
        atol=0.0,
    )
    assert first_cross.combined is None

    first_within = run("within-subject")
    second_within = run("within-subject")
    assert first_within.combined is not None
    assert second_within.combined is not None
    np.testing.assert_array_equal(
        first_within.combined.predictions,
        second_within.combined.predictions,
    )
    np.testing.assert_allclose(
        first_within.combined.probabilities,
        second_within.combined.probabilities,
        rtol=0.0,
        atol=0.0,
    )
