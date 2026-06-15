import numpy as np
import pytest

from experiments.logistic_regression import build_non_eeg_baselines


def test_builds_three_baselines_with_expected_contracts() -> None:
    y_train = np.asarray(
        [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int8,
    )

    baselines = build_non_eeg_baselines(y_train, n_test_samples=5, random_state=7)

    assert tuple(baseline.name for baseline in baselines) == (
        "global_majority",
        "pixel_frequency",
        "seeded_bernoulli",
    )
    assert all(baseline.probabilities.shape == (5, 3) for baseline in baselines)
    assert all(baseline.predictions.shape == (5, 3) for baseline in baselines)
    np.testing.assert_allclose(baselines[0].probabilities, y_train.mean())
    np.testing.assert_allclose(baselines[1].probabilities[0], [0.25, 0.75, 0.5])
    np.testing.assert_array_equal(baselines[1].predictions[0], [0, 1, 1])


def test_seeded_bernoulli_is_reproducible() -> None:
    y_train = np.asarray([[0, 1], [1, 0], [1, 1]], dtype=np.int8)

    first = build_non_eeg_baselines(y_train, n_test_samples=10, random_state=42)
    second = build_non_eeg_baselines(y_train, n_test_samples=10, random_state=42)

    np.testing.assert_array_equal(first[2].predictions, second[2].predictions)


@pytest.mark.parametrize(
    ("y_train", "n_test_samples", "threshold"),
    [
        ([], 1, 0.5),
        ([[0, 2]], 1, 0.5),
        ([[0, 1]], 0, 0.5),
        ([[0, 1]], 1, 1.0),
    ],
)
def test_rejects_invalid_baseline_inputs(
    y_train: object,
    n_test_samples: int,
    threshold: float,
) -> None:
    with pytest.raises(ValueError):
        build_non_eeg_baselines(
            y_train,
            n_test_samples=n_test_samples,
            threshold=threshold,
        )
