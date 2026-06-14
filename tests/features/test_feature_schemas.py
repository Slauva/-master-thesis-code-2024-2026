from pathlib import Path

import numpy as np
import pytest

from features import FeatureBlock, FeatureSet, flatten_feature_set
from utils.datasets import GeometricSample


def _sample() -> GeometricSample:
    return GeometricSample(
        subject_id=1,
        trial_number=2,
        Exec_Block_Index=3,
        eeg_path=Path("eeg.fif"),
        eog_path=Path("eog.fif"),
        img=[[0, 1], [1, 0]],
        pattern_id=4,
    )


def test_feature_set_accepts_modular_shapes() -> None:
    feature_set = FeatureSet(
        sample=_sample(),
        blocks=(
            FeatureBlock(
                name="time",
                layout="channel_features",
                values=np.ones((2, 2, 3), dtype=np.float32),
                feature_names=("mean", "variance", "rms"),
            ),
            FeatureBlock(
                name="covariance",
                layout="channel_matrix",
                values=np.stack([np.eye(2), np.eye(2)]).astype(np.float32),
            ),
            FeatureBlock(
                name="lndp",
                layout="channel_histogram",
                values=np.ones((2, 2, 4), dtype=np.float32),
                feature_names=("code_000", "code_001", "code_002", "code_003"),
            ),
        ),
        window_bounds_seconds=np.array([[0.5, 4.5], [2.5, 6.5]], dtype=np.float64),
        eeg_channels=("Fz", "Cz"),
        analysis_sfreq=125.0,
    )

    assert tuple(block.name for block in feature_set.blocks) == ("time", "covariance", "lndp")


def test_flatten_order_is_block_channel_then_feature() -> None:
    time_values = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    covariance = np.array([[[1.0, 2.0], [2.0, 3.0]]], dtype=np.float32)
    feature_set = FeatureSet(
        sample=_sample(),
        blocks=(
            FeatureBlock(
                name="time",
                layout="channel_features",
                values=time_values,
                feature_names=("mean", "variance"),
            ),
            FeatureBlock(
                name="covariance",
                layout="channel_matrix",
                values=covariance,
            ),
        ),
        window_bounds_seconds=np.array([[0.5, 15.5]], dtype=np.float64),
        eeg_channels=("Fz", "Cz"),
        analysis_sfreq=125.0,
    )

    matrix, names = flatten_feature_set(feature_set)

    np.testing.assert_allclose(matrix, [[1.0, 2.0, 3.0, 4.0, 1.0, 2.0 * np.sqrt(2.0), 3.0]])
    assert names == (
        "time:Fz:mean",
        "time:Fz:variance",
        "time:Cz:mean",
        "time:Cz:variance",
        "covariance:Fz:Fz",
        "covariance:Fz:Cz",
        "covariance:Cz:Cz",
    )


def test_flatten_supports_explicit_block_selection_and_order() -> None:
    feature_set = FeatureSet(
        sample=_sample(),
        blocks=(
            FeatureBlock(
                name="first",
                layout="channel_features",
                values=np.array([[[1.0], [2.0]]], dtype=np.float32),
                feature_names=("value",),
            ),
            FeatureBlock(
                name="second",
                layout="channel_features",
                values=np.array([[[3.0], [4.0]]], dtype=np.float32),
                feature_names=("value",),
            ),
        ),
        window_bounds_seconds=np.array([[0.5, 15.5]], dtype=np.float64),
        eeg_channels=("Fz", "Cz"),
        analysis_sfreq=125.0,
    )

    matrix, names = flatten_feature_set(feature_set, block_names=("second", "first"))

    np.testing.assert_array_equal(matrix, [[3.0, 4.0, 1.0, 2.0]])
    assert names[0].startswith("second:")
    assert names[2].startswith("first:")


@pytest.mark.parametrize(
    "block",
    [
        FeatureBlock(
            name="valid",
            layout="channel_features",
            values=np.ones((1, 2, 1), dtype=np.float32),
            feature_names=("value",),
        ),
    ],
)
def test_feature_set_rejects_channel_or_window_mismatch(block: FeatureBlock) -> None:
    with pytest.raises(ValueError, match="channel axis"):
        FeatureSet(
            sample=_sample(),
            blocks=(block,),
            window_bounds_seconds=np.array([[0.5, 15.5]], dtype=np.float64),
            eeg_channels=("Fp1", "Fz", "Cz"),
            analysis_sfreq=125.0,
        )

    with pytest.raises(ValueError, match="Window bounds"):
        FeatureSet(
            sample=_sample(),
            blocks=(block,),
            window_bounds_seconds=np.empty((0, 2), dtype=np.float64),
            eeg_channels=("Fz", "Cz"),
            analysis_sfreq=125.0,
        )


def test_feature_block_rejects_invalid_shape_names_or_values() -> None:
    with pytest.raises(ValueError, match="Unsupported feature block layout"):
        FeatureBlock(
            name="time",
            layout="invalid",  # type: ignore[arg-type]
            values=np.ones((1, 2, 1), dtype=np.float32),
            feature_names=("value",),
        )
    with pytest.raises(ValueError, match="three-dimensional"):
        FeatureBlock(
            name="time",
            layout="channel_features",
            values=np.ones((2, 3), dtype=np.float32),
            feature_names=("value",),
        )
    with pytest.raises(ValueError, match="Feature names"):
        FeatureBlock(
            name="time",
            layout="channel_features",
            values=np.ones((1, 2, 2), dtype=np.float32),
            feature_names=("value",),
        )
    with pytest.raises(ValueError, match="finite"):
        FeatureBlock(
            name="time",
            layout="channel_features",
            values=np.array([[[np.nan]]], dtype=np.float32),
            feature_names=("value",),
        )
