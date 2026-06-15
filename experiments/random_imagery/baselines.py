"""Shared non-EEG baselines for random-imagery reconstruction."""

import numpy as np
from numpy.typing import ArrayLike

from experiments.logistic_regression.schemas import BaselinePrediction


def build_non_eeg_baselines(
    y_train: ArrayLike,
    *,
    n_test_samples: int,
    threshold: float = 0.5,
    random_state: int = 42,
) -> tuple[BaselinePrediction, ...]:
    targets = np.asarray(y_train)
    if targets.ndim != 2 or targets.shape[0] < 1 or targets.shape[1] < 1:
        raise ValueError("`y_train` must have shape (sample, pixel) with non-empty axes")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("`y_train` must be binary")
    if isinstance(n_test_samples, bool) or n_test_samples < 1:
        raise ValueError("`n_test_samples` must be a positive integer")
    if not 0.0 < threshold < 1.0:
        raise ValueError("`threshold` must be between zero and one")

    n_pixels = targets.shape[1]
    global_frequency = float(targets.mean())
    pixel_frequencies = targets.mean(axis=0, dtype=np.float64)

    majority_probabilities = np.full(
        (n_test_samples, n_pixels),
        global_frequency,
        dtype=np.float64,
    )
    majority_predictions = np.full(
        (n_test_samples, n_pixels),
        int(global_frequency >= threshold),
        dtype=np.int8,
    )
    pixel_probabilities = np.broadcast_to(pixel_frequencies, (n_test_samples, n_pixels)).copy()
    pixel_predictions = (pixel_probabilities >= threshold).astype(np.int8)

    rng = np.random.default_rng(random_state)
    bernoulli_probabilities = pixel_probabilities.copy()
    bernoulli_predictions = (
        rng.random((n_test_samples, n_pixels)) < pixel_frequencies[np.newaxis, :]
    ).astype(np.int8)
    return (
        BaselinePrediction(
            name="global_majority",
            probabilities=majority_probabilities,
            predictions=majority_predictions,
        ),
        BaselinePrediction(
            name="pixel_frequency",
            probabilities=pixel_probabilities,
            predictions=pixel_predictions,
        ),
        BaselinePrediction(
            name="seeded_bernoulli",
            probabilities=bernoulli_probabilities,
            predictions=bernoulli_predictions,
        ),
    )
