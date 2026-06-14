import numpy as np
import pytest

from preprocessors import FFTConfig, compute_fft_psd, load_preprocessing_config


def test_fft_psd_localizes_channel_peaks_after_resampling() -> None:
    source_sfreq = 1_000.0
    time = np.arange(8_000, dtype=np.float64) / source_sfreq
    eeg = np.stack(
        (
            np.sin(2.0 * np.pi * 10.0 * time),
            0.5 * np.sin(2.0 * np.pi * 23.0 * time),
        )
    )
    config = load_preprocessing_config("fft")
    assert isinstance(config, FFTConfig)

    result = compute_fft_psd(eeg, source_sfreq=source_sfreq, config=config)

    assert result.eeg_power.shape == (2, 39)
    assert result.eeg_power.dtype == np.float32
    assert result.frequencies.dtype == np.float32
    assert result.times is None
    np.testing.assert_array_equal(
        result.frequencies,
        np.arange(2.0, 41.0, dtype=np.float32),
    )
    peak_frequencies = result.frequencies[np.argmax(result.eeg_power, axis=1)]
    np.testing.assert_array_equal(peak_frequencies, np.array([10.0, 23.0], dtype=np.float32))


def test_fft_psd_density_integrates_to_sine_mean_square() -> None:
    sfreq = 125.0
    time = np.arange(1_000, dtype=np.float64) / sfreq
    amplitude = 2.0
    eeg = amplitude * np.sin(2.0 * np.pi * 10.0 * time)[np.newaxis, :]
    config = load_preprocessing_config("fft")
    assert isinstance(config, FFTConfig)

    result = compute_fft_psd(eeg, source_sfreq=sfreq, config=config)

    integrated_power = result.eeg_power[0].sum() * config.frequency_step
    assert integrated_power == pytest.approx(amplitude**2 / 2.0, rel=1e-4)
    assert result.eeg_power[0, np.searchsorted(result.frequencies, 10.0)] == pytest.approx(
        amplitude**2 / 2.0,
        rel=1e-4,
    )


def test_fft_demeaning_removes_constant_offset_leakage() -> None:
    sfreq = 125.0
    time = np.arange(1_000, dtype=np.float64) / sfreq
    sine = np.sin(2.0 * np.pi * 10.0 * time)
    config = load_preprocessing_config("fft")
    assert isinstance(config, FFTConfig)

    baseline = compute_fft_psd(sine[np.newaxis, :], source_sfreq=sfreq, config=config)
    shifted = compute_fft_psd((sine + 100.0)[np.newaxis, :], source_sfreq=sfreq, config=config)

    np.testing.assert_allclose(shifted.eeg_power, baseline.eeg_power, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    ("eeg", "message"),
    [
        (np.ones(8), "shape"),
        (np.ones((2, 1)), "at least 2"),
        (np.array([[0.0, np.nan]]), "finite"),
    ],
)
def test_fft_rejects_invalid_signal(eeg: np.ndarray, message: str) -> None:
    config = load_preprocessing_config("fft")
    assert isinstance(config, FFTConfig)

    with pytest.raises(ValueError, match=message):
        compute_fft_psd(eeg, source_sfreq=125.0, config=config)
