from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import resample_poly


def prepare_eeg(
    eeg: ArrayLike,
    *,
    source_sfreq: float,
    target_sfreq: float,
    minimum_samples: int = 2,
) -> NDArray[np.float64]:
    signal = np.asarray(eeg, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError("`eeg` must have shape (channel, time)")
    if signal.shape[1] < minimum_samples:
        raise ValueError(f"EEG preprocessing requires at least {minimum_samples} time samples")
    if not np.isfinite(signal).all():
        raise ValueError("`eeg` must contain only finite values")
    if not np.isfinite(source_sfreq) or source_sfreq <= 0:
        raise ValueError("`source_sfreq` must be finite and positive")
    if not np.isfinite(target_sfreq) or target_sfreq <= 0:
        raise ValueError("`target_sfreq` must be finite and positive")
    if np.isclose(source_sfreq, target_sfreq, rtol=0.0, atol=1e-12):
        return signal

    ratio = (Fraction(str(target_sfreq)) / Fraction(str(source_sfreq))).limit_denominator(1_000_000)
    effective_sfreq = source_sfreq * ratio.numerator / ratio.denominator
    if not np.isclose(effective_sfreq, target_sfreq, rtol=1e-12, atol=1e-12):
        raise ValueError(
            f"Could not represent resampling ratio from {source_sfreq:g} Hz to {target_sfreq:g} Hz"
        )
    return resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1)
