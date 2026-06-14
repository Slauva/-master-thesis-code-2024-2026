import numpy as np
import pytest

from preprocessors import (
    MorletConfig,
    build_frequency_grid,
    build_morlet_cycles,
    compute_morlet_power,
    load_preprocessing_config,
)


def _burst(
    time: np.ndarray,
    *,
    frequency: float,
    start: float,
    stop: float,
) -> np.ndarray:
    mask = (time >= start) & (time < stop)
    envelope = np.zeros_like(time)
    envelope[mask] = np.hanning(mask.sum())
    return envelope * np.sin(2.0 * np.pi * frequency * time)


def test_builds_frequency_dependent_morlet_cycles() -> None:
    config = load_preprocessing_config("morlet")
    assert isinstance(config, MorletConfig)
    frequencies = build_frequency_grid(config)

    n_cycles = build_morlet_cycles(frequencies, config)

    np.testing.assert_array_equal(
        n_cycles,
        np.clip(frequencies / 2.0, 3.0, 10.0),
    )
    assert n_cycles[0] == 3.0
    assert n_cycles[-1] == 10.0


def test_morlet_power_localizes_synthetic_bursts_after_resampling() -> None:
    source_sfreq = 1_000.0
    time = np.arange(10_000, dtype=np.float64) / source_sfreq
    eeg = (
        _burst(time, frequency=10.0, start=2.0, stop=4.0)
        + _burst(time, frequency=25.0, start=6.0, stop=8.0)
    )[np.newaxis, :]
    config = load_preprocessing_config("morlet")
    assert isinstance(config, MorletConfig)

    result = compute_morlet_power(eeg, source_sfreq=source_sfreq, config=config)

    assert result.eeg_power.shape == (1, 39, 29)
    assert result.eeg_power.dtype == np.float32
    assert result.frequencies.dtype == np.float32
    assert result.times is not None
    assert result.times.dtype == np.float32
    assert result.scaling == "wavelet_power"
    np.testing.assert_allclose(np.diff(result.times), 0.256, rtol=0.0, atol=1e-6)

    for frequency, start, stop in ((10.0, 2.0, 4.0), (25.0, 6.0, 8.0)):
        frequency_index = int(np.searchsorted(result.frequencies, frequency))
        peak_time = float(result.times[np.argmax(result.eeg_power[0, frequency_index])])
        assert start <= peak_time <= stop

        burst_center = (start + stop) / 2.0
        time_index = int(np.argmin(np.abs(result.times - burst_center)))
        peak_frequency = float(result.frequencies[np.argmax(result.eeg_power[0, :, time_index])])
        assert peak_frequency == frequency


def test_morlet_trims_edges_and_centers_complete_time_bins() -> None:
    sfreq = 125.0
    time = np.arange(1_000, dtype=np.float64) / sfreq
    eeg = np.sin(2.0 * np.pi * 10.0 * time)[np.newaxis, :]
    config = load_preprocessing_config("morlet")
    assert isinstance(config, MorletConfig)

    result = compute_morlet_power(eeg, source_sfreq=sfreq, config=config)

    assert result.eeg_power.shape == (1, 39, 21)
    assert result.times is not None
    assert result.times[0] == pytest.approx(1.436)
    assert result.times[-1] == pytest.approx(6.556)
    assert result.times[0] > 149 / sfreq
    assert result.times[-1] < (time.size - 149) / sfreq


def test_morlet_rejects_signal_too_short_for_edges_and_time_bin() -> None:
    config = load_preprocessing_config("morlet")
    assert isinstance(config, MorletConfig)
    eeg = np.ones((2, 329), dtype=np.float64)

    with pytest.raises(ValueError, match="at least 330 resampled samples"):
        compute_morlet_power(eeg, source_sfreq=125.0, config=config)
