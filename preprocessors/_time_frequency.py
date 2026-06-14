import numpy as np
from numpy.typing import NDArray


def trim_and_bin_power(
    power: NDArray[np.float64],
    *,
    sfreq: float,
    edge_samples: int,
    bin_samples: int,
    method: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    valid_power = power[..., edge_samples : power.shape[-1] - edge_samples]
    n_bins = valid_power.shape[-1] // bin_samples
    if n_bins < 1:
        raise ValueError(f"{method} power does not contain a complete time bin after edge trimming")

    unused_samples = valid_power.shape[-1] - n_bins * bin_samples
    left_offset = unused_samples // 2
    first_sample = edge_samples + left_offset
    centered_power = valid_power[
        ...,
        left_offset : left_offset + n_bins * bin_samples,
    ]
    binned_power = centered_power.reshape(
        *centered_power.shape[:-1],
        n_bins,
        bin_samples,
    ).mean(axis=-1)
    times = (
        first_sample
        + np.arange(n_bins, dtype=np.float64) * bin_samples
        + (bin_samples - 1) / 2.0
    ) / sfreq
    return binned_power, times
