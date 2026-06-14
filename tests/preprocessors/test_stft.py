import numpy as np
import pytest

from preprocessors import (
    STFTConfig,
    build_short_time_fft,
    compute_stft_psd,
    load_preprocessing_config,
    minimum_unpadded_samples,
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


def test_stft_psd_localizes_synthetic_bursts_after_resampling() -> None:
    source_sfreq = 1_000.0
    time = np.arange(10_000, dtype=np.float64) / source_sfreq
    eeg = (
        _burst(time, frequency=10.0, start=2.0, stop=4.0)
        + _burst(time, frequency=25.0, start=6.0, stop=8.0)
    )[np.newaxis, :]
    config = load_preprocessing_config("stft")
    assert isinstance(config, STFTConfig)

    result = compute_stft_psd(eeg, source_sfreq=source_sfreq, config=config)

    assert result.eeg_power.shape == (1, 39, 32)
    assert result.eeg_power.dtype == np.float32
    assert result.frequencies.dtype == np.float32
    assert result.times is not None
    assert result.times.dtype == np.float32
    assert result.scaling == "psd"
    np.testing.assert_allclose(np.diff(result.times), 0.256, rtol=0.0, atol=1e-6)

    for frequency, start, stop in ((10.0, 2.0, 4.0), (25.0, 6.0, 8.0)):
        frequency_index = int(np.searchsorted(result.frequencies, frequency))
        peak_time = float(result.times[np.argmax(result.eeg_power[0, frequency_index])])
        assert start <= peak_time <= stop

        burst_center = (start + stop) / 2.0
        time_index = int(np.argmin(np.abs(result.times - burst_center)))
        peak_frequency = float(result.frequencies[np.argmax(result.eeg_power[0, :, time_index])])
        assert peak_frequency == frequency


def test_stft_psd_density_integrates_to_sine_mean_square() -> None:
    sfreq = 125.0
    time = np.arange(1_000, dtype=np.float64) / sfreq
    amplitude = 2.0
    eeg = amplitude * np.sin(2.0 * np.pi * 10.0 * time)[np.newaxis, :]
    config = load_preprocessing_config("stft")
    assert isinstance(config, STFTConfig)

    result = compute_stft_psd(eeg, source_sfreq=sfreq, config=config)

    integrated_power = result.eeg_power[0].sum(axis=0) * config.frequency_step
    np.testing.assert_allclose(integrated_power, amplitude**2 / 2.0, rtol=1e-5, atol=1e-6)


def test_stft_excludes_all_slices_affected_by_border_padding() -> None:
    sfreq = 125.0
    time = np.arange(1_000, dtype=np.float64) / sfreq
    eeg = np.sin(2.0 * np.pi * 10.0 * time)[np.newaxis, :]
    config = load_preprocessing_config("stft")
    assert isinstance(config, STFTConfig)
    transform = build_short_time_fft(config)
    p0 = transform.lower_border_end[1]
    p1 = transform.upper_border_begin(time.size)[1]

    result = compute_stft_psd(eeg, source_sfreq=sfreq, config=config)

    assert transform.m_num == 250
    assert transform.hop == 32
    assert transform.delta_f == 0.5
    assert (p0, p1) == (4, 28)
    assert result.eeg_power.shape == (1, 39, 24)
    assert result.times is not None
    np.testing.assert_allclose(result.times, transform.t(time.size, p0=p0, p1=p1))
    assert result.times[0] == pytest.approx(1.024)
    assert result.times[-1] == pytest.approx(6.912)


def test_stft_rejects_signal_too_short_for_an_unpadded_slice() -> None:
    config = load_preprocessing_config("stft")
    assert isinstance(config, STFTConfig)
    transform = build_short_time_fft(config)
    minimum_samples = minimum_unpadded_samples(transform)
    eeg = np.ones((2, minimum_samples - 1), dtype=np.float64)

    assert minimum_samples == 253
    with pytest.raises(ValueError, match="at least 253 resampled samples"):
        compute_stft_psd(eeg, source_sfreq=125.0, config=config)
