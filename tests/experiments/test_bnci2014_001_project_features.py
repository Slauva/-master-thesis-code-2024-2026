import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from experiments.bnci2014_001 import (
    BNCIProjectFeatureMatrix,
    BNCISplit,
    FeatureLogRegPrediction,
    build_epoch_dataset,
    create_leave_one_subject_splits,
    load_bnci_config,
    resolve_feature_benchmark_config,
    validate_baseline_manifest,
    validate_feature_alignment,
    write_feature_logreg_run,
)
from experiments.bnci2014_001 import workflow as bnci_workflow
from experiments.bnci2014_001.project_features import fit_predict_feature_logreg
from experiments.bnci2014_001.workflow import run_feature_logreg_loso
from features.config import build_feature_config_hash


def _toy_bnci_dataset() -> Any:
    rows = []
    labels = []
    for subject in (1, 2, 3):
        for label in ("left_hand", "right_hand", "feet", "tongue"):
            rows.append({"subject": subject, "session": "0train", "run": "0"})
            labels.append(label)
    X = np.zeros((len(rows), 4, 1001), dtype=np.float32)
    return build_epoch_dataset(X, labels, pd.DataFrame(rows))


def _toy_feature_matrix(config: Any, dataset: Any) -> BNCIProjectFeatureMatrix:
    feature_config = resolve_feature_benchmark_config(config.project_features)
    X = np.column_stack(
        (
            np.arange(dataset.y.shape[0], dtype=np.float32),
            np.asarray(dataset.y, dtype=np.float32),
            np.asarray(dataset.subjects, dtype=np.float32),
        )
    )
    X[:4] = 999.0
    X.setflags(write=False)
    y = np.asarray(dataset.y, dtype=np.int64)
    y.setflags(write=False)
    return BNCIProjectFeatureMatrix(
        X=X,
        y=y,
        sample_keys=dataset.sample_keys,
        feature_names=("row_index", "target_echo", "subject"),
        feature_config=feature_config,
        feature_config_hash=build_feature_config_hash(feature_config),
    )


def test_feature_logreg_fit_uses_train_rows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _toy_bnci_dataset()
    config = load_bnci_config(overrides={"dataset": {"subjects": [1, 2, 3]}})
    matrix = _toy_feature_matrix(config, dataset)
    split = create_leave_one_subject_splits(dataset)[0]
    calls: dict[str, np.ndarray] = {}

    class FakePipeline:
        def fit(self, X: np.ndarray, y: np.ndarray) -> "FakePipeline":
            calls["fit_X"] = X.copy()
            calls["fit_y"] = y.copy()
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            calls["predict_X"] = X.copy()
            return np.zeros(X.shape[0], dtype=np.int64)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.full((X.shape[0], 4), 0.25, dtype=np.float64)

    monkeypatch.setattr(
        "experiments.bnci2014_001.project_features._build_feature_logreg_pipeline",
        lambda _: FakePipeline(),
    )

    prediction = fit_predict_feature_logreg(
        matrix,
        split,
        dataset=dataset,
        benchmark_config=config.project_features,
        split_config=config.split,
    )

    assert prediction.y_pred.shape == prediction.y_true.shape
    assert not np.any(calls["fit_X"] == 999.0)
    assert np.all(calls["predict_X"] == 999.0)
    np.testing.assert_array_equal(calls["fit_y"], dataset.y[split.train_indices])


def test_feature_alignment_rejects_reordered_sample_keys() -> None:
    dataset = _toy_bnci_dataset()
    config = load_bnci_config(overrides={"dataset": {"subjects": [1, 2, 3]}})
    matrix = _toy_feature_matrix(config, dataset)
    reordered = BNCIProjectFeatureMatrix(
        X=matrix.X,
        y=matrix.y,
        sample_keys=tuple(reversed(matrix.sample_keys)),
        feature_names=matrix.feature_names,
        feature_config=matrix.feature_config,
        feature_config_hash=matrix.feature_config_hash,
    )

    with pytest.raises(ValueError, match="sample keys"):
        validate_feature_alignment(dataset, reordered)


def test_feature_workflow_writes_manifest_and_stage4_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _toy_bnci_dataset()
    config = load_bnci_config(
        overrides={
            "dataset": {"subjects": [1, 2, 3]},
            "artifacts": {"root": str(tmp_path / "runs")},
        }
    )
    matrix = _toy_feature_matrix(config, dataset)

    def fake_fit_predict(
        feature_matrix: BNCIProjectFeatureMatrix,
        split: BNCISplit,
        **_: object,
    ) -> Any:
        y_true = np.asarray(feature_matrix.y[split.test_indices], dtype=np.int64)
        return FeatureLogRegPrediction(
            split_name=split.name,
            y_true=y_true,
            y_pred=y_true.copy(),
            test_indices=np.asarray(split.test_indices, dtype=np.int64),
            probabilities=np.eye(4, dtype=np.float64)[y_true],
        )

    monkeypatch.setattr("experiments.bnci2014_001.workflow.fit_predict_feature_logreg", fake_fit_predict)

    result = run_feature_logreg_loso(config, dataset=dataset, feature_matrix=matrix)
    reference = {
        "run_dir": "artifacts/experiments/bnci2014_001/csp-lda/reference",
        "manifest": {"config_hash": "stage4reference"},
        "split": bnci_workflow._build_split_payload(result),  # noqa: SLF001
        "evaluation": {"summary": {"balanced_accuracy_mean": 0.25}},
    }
    run_dir = write_feature_logreg_run(config, result, reference=reference)

    validate_baseline_manifest(run_dir)
    assert (run_dir / "arrays" / "features.npy").is_file()
    comparison = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["split_alignment"] == "matched_by_fold_name_and_index_hash"
    assert comparison["balanced_accuracy_delta_vs_stage4"] == pytest.approx(0.75)

    with pytest.raises(FileExistsError):
        write_feature_logreg_run(config, result, reference=reference)
