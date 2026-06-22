import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from experiments.bnci2014_001 import (
    BNCISplit,
    build_epoch_dataset,
    create_leave_one_subject_splits,
    load_bnci_config,
    validate_baseline_manifest,
    validate_training_split,
    write_csp_lda_baseline_run,
)
from experiments.bnci2014_001.baselines import CSPBaselinePrediction, fit_predict_csp_lda
from experiments.bnci2014_001.metrics import evaluate_multiclass_predictions, summarize_fold_metrics
from experiments.bnci2014_001.workflow import run_csp_lda_loso


def _toy_bnci_dataset() -> Any:
    rows = []
    labels = []
    for subject in (1, 2, 3):
        for label in ("left_hand", "right_hand", "feet", "tongue"):
            rows.append({"subject": subject, "session": "0train", "run": "0"})
            labels.append(label)
    X = np.zeros((len(rows), 6, 32), dtype=np.float32)
    X[:4] = 999.0
    for index, label in enumerate(labels):
        X[index, index % 4] += 1.0
    return build_epoch_dataset(X, labels, pd.DataFrame(rows))


def test_multiclass_metrics_and_summary_payload_are_reproducible() -> None:
    y_true = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    y_pred = np.asarray([0, 1, 1, 1, 2, 0, 3, 2], dtype=np.int64)

    metrics = evaluate_multiclass_predictions(
        y_true,
        y_pred,
        class_names=("left_hand", "right_hand", "feet", "tongue"),
    )
    summary = summarize_fold_metrics((metrics,))

    assert metrics.n_samples == 8
    assert metrics.per_class_recall == {
        "left_hand": 0.5,
        "right_hand": 1.0,
        "feet": 0.5,
        "tongue": 0.5,
    }
    assert metrics.confusion_matrix.tolist() == [
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
    ]
    assert summary["balanced_accuracy_mean"] == pytest.approx(0.625)
    assert summary["n_folds"] == 1


def test_csp_lda_fit_uses_train_fold_only(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _toy_bnci_dataset()
    split = create_leave_one_subject_splits(dataset)[0]
    config = load_bnci_config(
        overrides={
            "dataset": {"subjects": [1, 2, 3]},
            "baseline": {"n_components": 2},
        }
    )
    calls: dict[str, np.ndarray] = {}

    class FakeCSP:
        def __init__(self, **kwargs: object) -> None:
            calls["n_components"] = np.asarray(kwargs["n_components"])

        def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
            calls["csp_fit_X"] = X.copy()
            calls["csp_fit_y"] = y.copy()
            return np.column_stack((y, np.arange(y.shape[0], dtype=np.float64)))

        def transform(self, X: np.ndarray) -> np.ndarray:
            calls["csp_test_X"] = X.copy()
            return np.zeros((X.shape[0], 2), dtype=np.float64)

    class FakeLDA:
        def __init__(self, **kwargs: object) -> None:
            calls["lda_solver"] = np.asarray(kwargs["solver"])

        def fit(self, X: np.ndarray, y: np.ndarray) -> "FakeLDA":
            calls["lda_fit_X"] = X.copy()
            calls["lda_fit_y"] = y.copy()
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0], dtype=np.int64)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.full((X.shape[0], 4), 0.25, dtype=np.float64)

    monkeypatch.setattr("experiments.bnci2014_001.baselines.CSP", FakeCSP)
    monkeypatch.setattr("experiments.bnci2014_001.baselines.LinearDiscriminantAnalysis", FakeLDA)

    prediction = fit_predict_csp_lda(
        dataset,
        split,
        baseline_config=config.baseline,
        split_config=config.split,
    )

    assert prediction.y_pred.shape == prediction.y_true.shape
    assert prediction.probabilities is not None
    assert prediction.probabilities.shape == (4, 4)
    assert not np.any(calls["csp_fit_X"] == 999.0)
    assert np.all(calls["csp_test_X"] >= 999.0)
    np.testing.assert_array_equal(calls["csp_fit_y"], dataset.y[split.train_indices])


def test_training_split_rejects_subject_leakage() -> None:
    dataset = _toy_bnci_dataset()
    split = BNCISplit(
        name="leaky",
        train_indices=np.asarray([4, 5, 6, 7], dtype=np.int64),
        test_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        train_subjects=(1, 2),
        test_subjects=(1,),
        n_samples=dataset.y.shape[0],
    )
    config = load_bnci_config(overrides={"dataset": {"subjects": [1, 2, 3]}})

    with pytest.raises(ValueError, match="forbidden leakage"):
        validate_training_split(dataset, split, split_config=config.split)


def test_workflow_writes_immutable_manifested_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _toy_bnci_dataset()
    config = load_bnci_config(
        overrides={
            "dataset": {"subjects": [1, 2, 3]},
            "baseline": {"n_components": 2},
            "artifacts": {"root": str(tmp_path / "runs")},
        }
    )

    def fake_fit_predict(
        dataset_arg: Any,
        split: BNCISplit,
        **_: object,
    ) -> CSPBaselinePrediction:
        y_true = np.asarray(dataset_arg.y[split.test_indices], dtype=np.int64)
        return CSPBaselinePrediction(
            split_name=split.name,
            y_true=y_true,
            y_pred=y_true.copy(),
            test_indices=np.asarray(split.test_indices, dtype=np.int64),
            probabilities=np.eye(4, dtype=np.float64)[y_true],
        )

    monkeypatch.setattr("experiments.bnci2014_001.workflow.fit_predict_csp_lda", fake_fit_predict)

    result = run_csp_lda_loso(config, dataset=dataset)
    run_dir = write_csp_lda_baseline_run(config, result)

    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "arrays" / "probabilities.npy").is_file()
    validate_baseline_manifest(run_dir)
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["summary"]["balanced_accuracy_mean"] == pytest.approx(1.0)

    with pytest.raises(FileExistsError):
        write_csp_lda_baseline_run(config, result)

    original_evaluation = (run_dir / "evaluation.json").read_text(encoding="utf-8")
    (run_dir / "evaluation.json").write_text(original_evaluation + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file size mismatch|file hash mismatch"):
        validate_baseline_manifest(run_dir)
