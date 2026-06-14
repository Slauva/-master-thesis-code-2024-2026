import numpy as np
import pytest

from features import build_window_layout, crop_eeg, load_feature_config


@pytest.mark.parametrize("n_times", [16_001, 26_001])
def test_default_crop_extracts_same_15_second_epoch(n_times: int) -> None:
    config = load_feature_config()
    eeg = np.arange(2 * n_times, dtype=np.float32).reshape(2, n_times)

    cropped = crop_eeg(eeg, source_sfreq=1_000.0, config=config)

    assert cropped.source_slice == slice(500, 15_500)
    assert cropped.bounds_seconds == (0.5, 15.5)
    assert cropped.eeg.shape == (2, 15_000)
    np.testing.assert_array_equal(cropped.eeg, eeg[:, 500:15_500])


def test_default_layout_contains_one_full_epoch_window() -> None:
    config = load_feature_config()

    layout = build_window_layout(n_times=1_875, config=config)

    assert layout.slices == (slice(0, 1_875),)
    np.testing.assert_array_equal(layout.bounds_seconds, np.array([[0.5, 15.5]]))


def test_configured_layout_retains_only_complete_windows() -> None:
    config = load_feature_config(
        overrides={
            "window_seconds": 4.0,
            "window_stride_seconds": 2.0,
        }
    )

    layout = build_window_layout(n_times=1_875, config=config)

    assert layout.slices == (
        slice(0, 500),
        slice(250, 750),
        slice(500, 1_000),
        slice(750, 1_250),
        slice(1_000, 1_500),
        slice(1_250, 1_750),
    )
    np.testing.assert_array_equal(
        layout.bounds_seconds,
        np.array(
            [
                [0.5, 4.5],
                [2.5, 6.5],
                [4.5, 8.5],
                [6.5, 10.5],
                [8.5, 12.5],
                [10.5, 14.5],
            ]
        ),
    )


def test_crop_rejects_short_or_non_finite_signal() -> None:
    config = load_feature_config()

    with pytest.raises(ValueError, match="contain only"):
        crop_eeg(np.array([[0.0, np.nan]], dtype=np.float32), source_sfreq=1_000.0, config=config)
    with pytest.raises(ValueError, match="only 10"):
        crop_eeg(np.zeros((2, 10_000), dtype=np.float32), source_sfreq=1_000.0, config=config)


def test_crop_requires_exact_source_sample_boundaries() -> None:
    config = load_feature_config()

    with pytest.raises(ValueError, match="integer sample"):
        crop_eeg(np.zeros((2, 2_000), dtype=np.float32), source_sfreq=127.0, config=config)


def test_layout_rejects_unexpected_analysis_epoch_length() -> None:
    config = load_feature_config()

    with pytest.raises(ValueError, match="expected 1875"):
        build_window_layout(n_times=1_874, config=config)
