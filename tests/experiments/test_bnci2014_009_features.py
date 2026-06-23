import numpy as np
import pandas as pd
import pytest

from experiments.bnci2014_009 import build_epoch_dataset
from experiments.bnci2014_009.features import (
    build_erp_feature_matrix,
    fit_transform_xdawn_tangent_space,
)


def _toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for subject in (1, 2):
        for index in range(12):
            rows.append({"subject": subject, "session": "0", "run": "0"})
            labels.append("Target" if index % 6 == 0 else "NonTarget")
    X = np.linspace(-1.0, 1.0, num=len(rows) * 4 * 32, dtype=np.float64).reshape(len(rows), 4, 32)
    dataset = build_epoch_dataset(
        X,
        labels,
        pd.DataFrame(rows),
        dtype="float32",
    )
    return dataset.X, dataset.y


def test_build_erp_feature_matrix_combines_waveform_and_window_features() -> None:
    X, _y = _toy_dataset()
    metadata = pd.DataFrame(
        {"subject": [1] * X.shape[0], "session": ["0"] * X.shape[0], "run": ["0"] * X.shape[0]}
    )
    dataset = build_epoch_dataset(
        X,
        ["Target" if index % 6 == 0 else "NonTarget" for index in range(X.shape[0])],
        metadata,
    )

    matrix = build_erp_feature_matrix(
        dataset,
        source_sfreq=32.0,
        waveform_stride=4,
        window_bounds_seconds=((0.0, 0.5), (0.5, 1.0)),
        channel_names=("C1", "C2", "C3", "C4"),
    )

    assert matrix.X.shape == (24, 40)
    assert matrix.y.dtype == np.int64
    assert matrix.X.dtype == np.float32
    assert matrix.feature_names[0] == "waveform:C1:sample_0:t_0.000000s"
    assert matrix.feature_names[7] == "waveform:C1:sample_28:t_0.875000s"
    assert matrix.feature_names[-1] == "window_mean:C4:0.500-1.000s"
    assert matrix.waveform_time_indices == (0, 4, 8, 12, 16, 20, 24, 28)
    assert matrix.window_bounds_seconds == ((0.0, 0.5), (0.5, 1.0))
    assert matrix.sample_keys == dataset.sample_keys
    assert not matrix.X.flags.writeable


def test_build_erp_feature_matrix_rejects_bad_shape_parameters() -> None:
    X, _y = _toy_dataset()
    dataset = build_epoch_dataset(
        X,
        ["Target" if index % 6 == 0 else "NonTarget" for index in range(X.shape[0])],
        pd.DataFrame({"subject": [1] * X.shape[0], "session": ["0"] * X.shape[0], "run": ["0"] * X.shape[0]}),
    )

    with pytest.raises(ValueError, match="waveform_stride"):
        build_erp_feature_matrix(dataset, source_sfreq=32.0, waveform_stride=0, channel_names=("C1",) * 4)

    with pytest.raises(ValueError, match="channel_names"):
        build_erp_feature_matrix(dataset, source_sfreq=32.0, channel_names=("C1", "C2"))

    with pytest.raises(ValueError, match="Window bounds"):
        build_erp_feature_matrix(
            dataset,
            source_sfreq=32.0,
            channel_names=("C1",) * 4,
            window_bounds_seconds=((0.3, 0.2),),
        )


def test_fit_transform_xdawn_tangent_space_is_explicit_train_apply_boundary() -> None:
    X, y = _toy_dataset()
    train_indices = np.arange(0, 18, dtype=np.int64)
    apply_indices = np.arange(18, 24, dtype=np.int64)

    features = fit_transform_xdawn_tangent_space(
        X[train_indices],
        y[train_indices],
        X[apply_indices],
        n_filters=1,
        estimator="oas",
    )

    assert features.X_train.shape[0] == train_indices.size
    assert features.X_apply.shape[0] == apply_indices.size
    assert features.X_train.shape[1] == features.X_apply.shape[1]
    assert features.train_covariances_shape[0] == train_indices.size
    assert features.apply_covariances_shape[0] == apply_indices.size
    assert features.n_filters == 1
    assert features.estimator == "oas"
    assert not features.X_train.flags.writeable
    assert not features.X_apply.flags.writeable


def test_fit_transform_xdawn_tangent_space_rejects_misaligned_targets() -> None:
    X, y = _toy_dataset()

    with pytest.raises(ValueError, match="align"):
        fit_transform_xdawn_tangent_space(X[:6], y[:5], X[6:8])

    with pytest.raises(ValueError, match="at least two classes"):
        fit_transform_xdawn_tangent_space(X[:5], np.ones(5, dtype=np.int64), X[6:8])
