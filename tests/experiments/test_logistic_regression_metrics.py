from pathlib import Path

import numpy as np
import pytest

from experiments.logistic_regression import (
    bootstrap_subject_mean_balanced_accuracy,
    build_non_eeg_baselines,
    evaluate_prediction_matrix,
)


def test_prediction_metrics_match_known_binary_matrix() -> None:
    targets = np.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int8,
    )
    predictions = np.asarray(
        [
            [0, 0],
            [1, 1],
            [1, 1],
            [1, 0],
        ],
        dtype=np.int8,
    )
    probabilities = np.asarray(
        [
            [0.1, 0.2],
            [0.8, 0.9],
            [0.9, 0.7],
            [0.8, 0.4],
        ],
        dtype=np.float64,
    )

    metrics = evaluate_prediction_matrix(targets, predictions, probabilities)

    np.testing.assert_allclose(metrics.per_pixel_balanced_accuracy, [0.75, 0.5])
    assert metrics.mean_balanced_accuracy == pytest.approx(0.625)
    assert metrics.bit_accuracy == pytest.approx(0.625)
    assert metrics.exact_match_accuracy == pytest.approx(0.25)
    assert metrics.mean_hamming_distance == pytest.approx(0.75)
    assert metrics.hamming_loss == pytest.approx(0.375)
    np.testing.assert_allclose(metrics.per_sample_iou, [1.0, 0.5, 0.5, 0.5])
    assert metrics.mean_sample_iou == pytest.approx(0.625)
    assert metrics.micro_iou == pytest.approx(3.0 / 6.0)
    assert metrics.mean_macro_f1 == pytest.approx(
        metrics.per_pixel_macro_f1.mean()
    )
    assert metrics.mean_brier_score == pytest.approx(
        metrics.per_pixel_brier_score.mean()
    )


def test_iou_handles_perfect_partial_disjoint_and_empty_samples() -> None:
    targets = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int8,
    )
    predictions = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int8,
    )

    metrics = evaluate_prediction_matrix(
        targets,
        predictions,
        probabilities=predictions,
    )

    np.testing.assert_allclose(
        metrics.per_sample_iou,
        [1.0, 1.0 / 3.0, 0.0, 1.0, 1.0],
    )
    assert metrics.mean_sample_iou == pytest.approx(2.0 / 3.0)
    assert metrics.micro_iou == pytest.approx(5.0 / 11.0)
    assert metrics.hamming_loss == pytest.approx(1.0 - metrics.bit_accuracy)
    assert metrics.hamming_loss == pytest.approx(
        metrics.mean_hamming_distance / targets.shape[1]
    )


def test_reference_run_reconstruction_metrics_are_stable() -> None:
    run_dir = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "experiments"
        / "logistic-regression"
        / "f515948b6bf5af55"
    )
    arrays_dir = run_dir / "arrays"
    targets = np.load(arrays_dir / "test_targets.npy", allow_pickle=False)
    predictions = np.load(arrays_dir / "predictions.npy", allow_pickle=False)
    probabilities = np.load(arrays_dir / "probabilities.npy", allow_pickle=False)
    train_targets = np.load(arrays_dir / "train_targets.npy", allow_pickle=False)

    model = evaluate_prediction_matrix(targets, predictions, probabilities)
    assert model.mean_sample_iou == pytest.approx(0.335257970, abs=5e-10)
    assert model.micro_iou == pytest.approx(0.334634146, abs=5e-10)
    assert model.hamming_loss == pytest.approx(0.485754986, abs=5e-10)

    baselines = build_non_eeg_baselines(
        train_targets,
        n_test_samples=targets.shape[0],
        random_state=42,
    )
    for baseline in baselines:
        metrics = evaluate_prediction_matrix(
            targets,
            baseline.predictions,
            baseline.probabilities,
        )
        assert metrics.hamming_loss == pytest.approx(1.0 - metrics.bit_accuracy)
        assert metrics.hamming_loss == pytest.approx(
            metrics.mean_hamming_distance / targets.shape[1]
        )


def test_subject_bootstrap_is_deterministic_and_clustered() -> None:
    subject_ids = np.repeat(np.arange(1, 7, dtype=np.int64), 2)
    targets = np.tile(
        np.asarray([[0, 1], [1, 0]], dtype=np.int8),
        (6, 1),
    )
    predictions = targets.copy()
    predictions[subject_ids == 6, 0] = 1 - predictions[subject_ids == 6, 0]

    first = bootstrap_subject_mean_balanced_accuracy(
        targets,
        predictions,
        subject_ids,
        n_resamples=200,
        random_state=42,
    )
    second = bootstrap_subject_mean_balanced_accuracy(
        targets,
        predictions,
        subject_ids,
        n_resamples=200,
        random_state=42,
    )

    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.estimate == pytest.approx(11.0 / 12.0)
    assert first.lower <= first.estimate <= first.upper
    assert first.n_attempts == first.n_resamples


@pytest.mark.parametrize(
    ("targets", "predictions", "probabilities"),
    [
        ([[0, 1]], [[0, 1]], [[0.1, 0.9]]),
        ([[0, 1], [1, 0]], [[0, 2], [1, 0]], [[0.1, 0.9], [0.9, 0.1]]),
        ([[0, 1], [1, 0]], [[0, 1], [1, 0]], [[0.1], [0.9]]),
    ],
)
def test_prediction_metrics_reject_invalid_inputs(
    targets: object,
    predictions: object,
    probabilities: object,
) -> None:
    with pytest.raises(ValueError):
        evaluate_prediction_matrix(targets, predictions, probabilities)
