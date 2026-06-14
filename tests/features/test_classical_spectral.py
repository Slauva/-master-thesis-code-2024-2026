import numpy as np
import pytest

from features import PreparedFeatureWindows, extract_spectral_features, load_feature_config


def test_spectral_features_localize_tones_and_preserve_power() -> None:
    sfreq = 125.0
    times = np.arange(5 * int(sfreq), dtype=np.float64) / sfreq
    signals = np.stack(
        (
            2.0 * np.sin(2.0 * np.pi * 10.0 * times),
            np.sin(2.0 * np.pi * 20.0 * times),
        )
    )
    windows = PreparedFeatureWindows(
        values=signals[np.newaxis, ...],
        bounds_seconds=np.array([[0.5, 5.5]], dtype=np.float64),
        sfreq=sfreq,
    )

    block = extract_spectral_features(windows, config=load_feature_config())
    channel_features = [
        dict(zip(block.feature_names, block.values[0, channel], strict=True))
        for channel in range(2)
    ]

    ten_hz = channel_features[0]
    assert ten_hz["absolute_band_power_alpha"] == pytest.approx(2.0, rel=0.02)
    assert ten_hz["relative_band_power_alpha"] == pytest.approx(1.0, abs=1e-5)
    assert ten_hz["total_power"] == pytest.approx(2.0, rel=0.02)
    assert ten_hz["dominant_frequency"] == pytest.approx(10.0)
    assert ten_hz["spectral_centroid"] == pytest.approx(10.0, abs=0.05)

    twenty_hz = channel_features[1]
    assert twenty_hz["absolute_band_power_beta"] == pytest.approx(0.5, rel=0.02)
    assert twenty_hz["relative_band_power_beta"] == pytest.approx(1.0, abs=1e-5)
    assert twenty_hz["dominant_frequency"] == pytest.approx(20.0)
    assert twenty_hz["spectral_centroid"] == pytest.approx(20.0, abs=0.05)
    assert 0.0 <= twenty_hz["spectral_entropy"] <= 1.0


def test_zero_signal_spectral_summaries_are_finite_zeroes() -> None:
    windows = PreparedFeatureWindows(
        values=np.zeros((1, 2, 500), dtype=np.float64),
        bounds_seconds=np.array([[0.5, 4.5]], dtype=np.float64),
        sfreq=125.0,
    )

    block = extract_spectral_features(windows, config=load_feature_config())

    np.testing.assert_array_equal(block.values, np.zeros_like(block.values))
