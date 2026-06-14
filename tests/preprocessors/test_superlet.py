import numpy as np
import pytest

from preprocessors import (
    SuperletConfig,
    build_frequency_grid,
    compute_adaptive_order,
    compute_superlet_power,
    load_preprocessing_config,
    superlet_edge_samples,
)


def test_builds_linearly_increasing_adaptive_orders() -> None:
    config = load_preprocessing_config("superlet")
    assert isinstance(config, SuperletConfig)
    frequencies = build_frequency_grid(config)

    orders = compute_adaptive_order(
        frequencies,
        config.order_min,
        config.order_max,
    )

    assert orders[0] == config.order_min
    assert orders[-1] == config.order_max
    assert np.all(np.diff(orders) > 0)
    np.testing.assert_allclose(np.diff(orders), np.diff(orders)[0])


def test_superlet_power_separates_close_frequencies() -> None:
    source_sfreq = 1_000.0
    time = np.arange(10_000, dtype=np.float64) / source_sfreq
    eeg = (
        np.sin(2.0 * np.pi * 20.0 * time)
        + np.sin(2.0 * np.pi * 24.0 * time)
    )[np.newaxis, :]
    config = load_preprocessing_config("superlet")
    assert isinstance(config, SuperletConfig)

    result = compute_superlet_power(eeg, source_sfreq=source_sfreq, config=config)

    assert result.eeg_power.shape == (1, 39, 26)
    assert result.eeg_power.dtype == np.float32
    assert result.frequencies.dtype == np.float32
    assert result.times is not None
    assert result.times.dtype == np.float32
    assert result.scaling == "wavelet_power"
    assert np.isfinite(result.eeg_power).all()
    assert np.all(result.eeg_power >= 0)
    np.testing.assert_allclose(np.diff(result.times), 0.256, rtol=0.0, atol=1e-6)

    spectrum = result.eeg_power[0].mean(axis=1)
    frequency_indices = {
        frequency: int(np.searchsorted(result.frequencies, frequency))
        for frequency in (19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0)
    }
    assert spectrum[frequency_indices[20.0]] > spectrum[frequency_indices[19.0]]
    assert spectrum[frequency_indices[20.0]] > spectrum[frequency_indices[21.0]]
    assert spectrum[frequency_indices[24.0]] > spectrum[frequency_indices[23.0]]
    assert spectrum[frequency_indices[24.0]] > spectrum[frequency_indices[25.0]]
    assert spectrum[frequency_indices[22.0]] < min(
        spectrum[frequency_indices[20.0]],
        spectrum[frequency_indices[24.0]],
    )


def test_superlet_trims_longest_fractional_wavelet_and_centers_bins() -> None:
    sfreq = 125.0
    time = np.arange(1_000, dtype=np.float64) / sfreq
    eeg = np.sin(2.0 * np.pi * 10.0 * time)[np.newaxis, :]
    config = load_preprocessing_config("superlet")
    assert isinstance(config, SuperletConfig)
    frequencies = build_frequency_grid(config)

    edge_samples = superlet_edge_samples(
        frequencies,
        sfreq=sfreq,
        order_min=config.order_min,
        order_max=config.order_max,
        c_1=config.c_1,
    )
    result = compute_superlet_power(eeg, source_sfreq=sfreq, config=config)

    assert edge_samples == 199
    assert result.eeg_power.shape == (1, 39, 18)
    assert result.times is not None
    assert result.times[0] == pytest.approx(1.820)
    assert result.times[-1] == pytest.approx(6.172)
    assert result.times[0] > edge_samples / sfreq
    assert result.times[-1] < (time.size - edge_samples) / sfreq


def test_superlet_rejects_signal_too_short_for_edges_and_time_bin() -> None:
    config = load_preprocessing_config("superlet")
    assert isinstance(config, SuperletConfig)
    eeg = np.ones((2, 429), dtype=np.float64)

    with pytest.raises(ValueError, match="at least 430 resampled samples"):
        compute_superlet_power(eeg, source_sfreq=125.0, config=config)
