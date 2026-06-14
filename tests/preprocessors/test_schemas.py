import numpy as np
import pytest

from preprocessors import SpectralTransformResult


def test_accepts_frequency_only_and_time_frequency_power() -> None:
    frequency_only = SpectralTransformResult(
        eeg_power=np.ones((2, 3)),
        frequencies=np.array([2.0, 3.0, 4.0]),
        times=None,
        analysis_sfreq=125.0,
        scaling="psd",
    )
    time_frequency = SpectralTransformResult(
        eeg_power=np.ones((2, 3, 4)),
        frequencies=np.array([2.0, 3.0, 4.0]),
        times=np.array([0.0, 0.25, 0.5, 0.75]),
        analysis_sfreq=125.0,
        scaling="wavelet_power",
    )

    assert frequency_only.eeg_power.shape == (2, 3)
    assert time_frequency.eeg_power.shape == (2, 3, 4)


@pytest.mark.parametrize(
    ("power", "frequencies", "times", "message"),
    [
        (np.ones((2, 3, 4)), np.array([2.0, 3.0, 4.0]), None, "requires a time axis"),
        (np.ones((2, 2)), np.array([2.0, 3.0, 4.0]), None, "frequency axis"),
        (np.array([[1.0, -1.0]]), np.array([2.0, 3.0]), None, "non-negative"),
        (np.ones((1, 2)), np.array([3.0, 2.0]), None, "strictly increasing"),
    ],
)
def test_rejects_invalid_transform_results(
    power: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SpectralTransformResult(
            eeg_power=power,
            frequencies=frequencies,
            times=times,
            analysis_sfreq=125.0,
            scaling="psd",
        )
