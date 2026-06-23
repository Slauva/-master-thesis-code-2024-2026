import numpy as np
import pytest

from experiments.random_imagery.full_sweep_comparison import (
    _build_subject_bootstrap_rows,
    _mean_balanced_accuracy,
    validate_full_imagery_comparison_summary,
)
from experiments.random_imagery.metrics import evaluate_prediction_matrix


def test_mean_balanced_accuracy_matches_shared_metric() -> None:
    targets = np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.int8,
    )
    predictions = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )
    scores = predictions.astype(np.float64)

    assert _mean_balanced_accuracy(targets, predictions) == pytest.approx(
        evaluate_prediction_matrix(targets, predictions, scores).mean_balanced_accuracy
    )


def test_subject_bootstrap_rows_are_class_complete() -> None:
    targets = np.array(
        [
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
        ],
        dtype=np.int8,
    )
    subject_ids = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)

    rows, attempts = _build_subject_bootstrap_rows(
        targets,
        subject_ids,
        n_resamples=8,
        random_state=11,
    )

    assert len(rows) == 8
    assert attempts >= len(rows)
    for indices in rows:
        sampled = targets[indices]
        assert np.all(sampled.sum(axis=0) > 0)
        assert np.all(sampled.sum(axis=0) < sampled.shape[0])


def test_full_imagery_summary_validation_rejects_changed_coverage() -> None:
    payload = {
        "schema_version": 1,
        "coverage": {
            "classical": {"completed_protocol_run_count": 166},
            "torch": {"completed_protocol_run_count": 24},
            "failed_protocol_run_count": 13,
        },
        "protocol_summaries": {},
    }

    with pytest.raises(ValueError, match="Classical completed protocol count"):
        validate_full_imagery_comparison_summary(payload)
