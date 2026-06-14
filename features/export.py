from typing import Protocol

import numpy as np

from features.schemas import FeatureMatrix, FeatureSet, RecordingFamily, SampleKey, flatten_feature_set


class FeatureSetDataset(Protocol):
    dataset_step_type: RecordingFamily

    def __len__(self) -> int: ...

    def __getitem__(self, key: int | SampleKey) -> FeatureSet: ...


def build_feature_matrix(
    dataset: FeatureSetDataset,
    *,
    block_names: tuple[str, ...] | None = None,
) -> FeatureMatrix:
    """Flatten one recording-family dataset without fitting learned transforms."""
    if len(dataset) < 1:
        raise ValueError("Cannot build a feature matrix from an empty dataset")
    if dataset.dataset_step_type not in ("exec", "patt"):
        raise ValueError(f"Unsupported recording family: {dataset.dataset_step_type!r}")

    matrices: list[np.ndarray] = []
    expected_names: tuple[str, ...] | None = None
    sample_keys: list[SampleKey] = []
    window_indices: list[np.ndarray] = []
    window_bounds: list[np.ndarray] = []

    for index in range(len(dataset)):
        feature_set = dataset[index]
        matrix, names = flatten_feature_set(feature_set, block_names=block_names)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError("Feature names or EEG channel order differ between dataset samples")

        n_windows = matrix.shape[0]
        key = (
            feature_set.sample.subject_id,
            feature_set.sample.trial_number,
            feature_set.sample.block_index,
        )
        matrices.append(matrix)
        sample_keys.extend([key] * n_windows)
        window_indices.append(np.arange(n_windows, dtype=np.int64))
        window_bounds.append(feature_set.window_bounds_seconds)

    if expected_names is None:
        raise RuntimeError("Feature matrix construction produced no feature names")
    return FeatureMatrix(
        X=np.concatenate(matrices, axis=0),
        feature_names=expected_names,
        sample_keys=tuple(sample_keys),
        window_indices=np.concatenate(window_indices),
        window_bounds_seconds=np.concatenate(window_bounds, axis=0),
        recording_family=dataset.dataset_step_type,
    )
