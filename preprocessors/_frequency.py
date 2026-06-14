import numpy as np
from numpy.typing import NDArray


def rebin_density(
    psd: NDArray[np.float64],
    *,
    source_frequencies: NDArray[np.float64],
    target_frequencies: NDArray[np.float64],
    target_width: float,
) -> NDArray[np.float64]:
    if source_frequencies.size < 2:
        raise ValueError("PSD rebinning requires at least two source frequency bins")

    source_edges = _frequency_edges(source_frequencies)
    target_edges = np.concatenate(
        (
            target_frequencies - target_width / 2.0,
            np.array([target_frequencies[-1] + target_width / 2.0]),
        )
    )
    overlap = np.maximum(
        0.0,
        np.minimum(target_edges[1:, np.newaxis], source_edges[np.newaxis, 1:])
        - np.maximum(target_edges[:-1, np.newaxis], source_edges[np.newaxis, :-1]),
    )
    covered_width = overlap.sum(axis=1)
    if not np.allclose(covered_width, target_width):
        raise ValueError("The native frequency grid does not fully cover the configured output bins")
    return psd @ overlap.T / target_width


def _frequency_edges(frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
    midpoints = (frequencies[:-1] + frequencies[1:]) / 2.0
    first = frequencies[0] - (midpoints[0] - frequencies[0])
    last = frequencies[-1] + (frequencies[-1] - midpoints[-1])
    return np.concatenate((np.array([first]), midpoints, np.array([last])))
