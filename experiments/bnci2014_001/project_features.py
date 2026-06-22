from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.bnci2014_001.baselines import validate_training_split
from experiments.bnci2014_001.config import (
    BNCIProjectFeatureBenchmarkConfig,
    BNCISplitConfig,
)
from experiments.bnci2014_001.data import BNCIEpochDataset, BNCISampleKey, BNCISplit
from experiments.bnci2014_001.features import (
    extract_bnci_feature_set,
    flatten_bnci_feature_set,
    load_bnci_feature_config,
)
from features.config import FeatureExtractionConfig, build_feature_config_hash

FEATURE_LOGREG_VERSION = 1


@dataclass(frozen=True, slots=True)
class BNCIProjectFeatureMatrix:
    X: NDArray[np.floating[Any]]
    y: NDArray[np.int64]
    sample_keys: tuple[BNCISampleKey, ...]
    feature_names: tuple[str, ...]
    feature_config: FeatureExtractionConfig
    feature_config_hash: str

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("Project feature matrix must be two-dimensional")
        if self.y.ndim != 1:
            raise ValueError("Project feature targets must be one-dimensional")
        if self.X.shape[0] != self.y.shape[0] or self.X.shape[0] != len(self.sample_keys):
            raise ValueError("Project feature rows, targets, and sample keys must align")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError("Project feature columns must align with feature names")
        if len(set(self.sample_keys)) != len(self.sample_keys):
            raise ValueError("Project feature matrix must contain one row per BNCI epoch")
        if not np.isfinite(self.X).all():
            raise ValueError("Project feature matrix must contain only finite values")


@dataclass(frozen=True, slots=True)
class FeatureLogRegPrediction:
    split_name: str
    y_true: NDArray[np.int64]
    y_pred: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    probabilities: NDArray[np.float64]


def resolve_feature_benchmark_config(
    benchmark_config: BNCIProjectFeatureBenchmarkConfig,
) -> FeatureExtractionConfig:
    return load_bnci_feature_config(
        overrides={
            "feature_groups": list(benchmark_config.feature_groups),
            "window_seconds": benchmark_config.window_seconds,
            "window_stride_seconds": benchmark_config.window_stride_seconds,
        }
    )


def build_project_feature_matrix(
    dataset: BNCIEpochDataset,
    *,
    benchmark_config: BNCIProjectFeatureBenchmarkConfig,
    source_sfreq: float,
) -> BNCIProjectFeatureMatrix:
    feature_config = resolve_feature_benchmark_config(benchmark_config)
    rows: list[NDArray[np.floating[Any]]] = []
    sample_keys: list[BNCISampleKey] = []
    feature_names: tuple[str, ...] | None = None

    for epoch, metadata in zip(dataset.X, dataset.metadata, strict=True):
        feature_set = extract_bnci_feature_set(
            epoch,
            metadata,
            config=feature_config,
            source_sfreq=source_sfreq,
        )
        matrix = flatten_bnci_feature_set(feature_set)
        if matrix.X.shape[0] != 1:
            raise ValueError(
                "Stage 5 feature benchmark requires exactly one feature row per BNCI epoch; "
                "use window_seconds=null and window_stride_seconds=null"
            )
        if feature_names is None:
            feature_names = matrix.feature_names
        elif matrix.feature_names != feature_names:
            raise ValueError("Feature names changed while materializing BNCI project features")
        rows.append(matrix.X[0])
        sample_keys.append(metadata.sample_key)

    X = np.asarray(np.vstack(rows), dtype=np.float32)
    X.setflags(write=False)
    y = np.asarray(dataset.y, dtype=np.int64)
    y.setflags(write=False)
    return BNCIProjectFeatureMatrix(
        X=X,
        y=y,
        sample_keys=tuple(sample_keys),
        feature_names=feature_names or (),
        feature_config=feature_config,
        feature_config_hash=build_feature_config_hash(feature_config),
    )


def fit_predict_feature_logreg(
    feature_matrix: BNCIProjectFeatureMatrix,
    split: BNCISplit,
    *,
    dataset: BNCIEpochDataset,
    benchmark_config: BNCIProjectFeatureBenchmarkConfig,
    split_config: BNCISplitConfig,
) -> FeatureLogRegPrediction:
    validate_feature_alignment(dataset, feature_matrix)
    validate_training_split(dataset, split, split_config=split_config)

    train_indices = np.asarray(split.train_indices, dtype=np.int64)
    test_indices = np.asarray(split.test_indices, dtype=np.int64)
    X_train = np.asarray(feature_matrix.X[train_indices], dtype=np.float64)
    y_train = np.asarray(feature_matrix.y[train_indices], dtype=np.int64)
    X_test = np.asarray(feature_matrix.X[test_indices], dtype=np.float64)
    y_test = np.asarray(feature_matrix.y[test_indices], dtype=np.int64)

    pipeline = _build_feature_logreg_pipeline(benchmark_config)
    pipeline.fit(X_train, y_train)
    y_pred = np.asarray(pipeline.predict(X_test), dtype=np.int64)
    probabilities = np.asarray(pipeline.predict_proba(X_test), dtype=np.float64)
    if probabilities.shape != (test_indices.size, len(dataset.class_names)):
        raise ValueError("Feature Logistic Regression probabilities have an unexpected shape")

    y_test.setflags(write=False)
    y_pred.setflags(write=False)
    test_indices.setflags(write=False)
    probabilities.setflags(write=False)
    return FeatureLogRegPrediction(
        split_name=split.name,
        y_true=y_test,
        y_pred=y_pred,
        test_indices=test_indices,
        probabilities=probabilities,
    )


def validate_feature_alignment(
    dataset: BNCIEpochDataset,
    feature_matrix: BNCIProjectFeatureMatrix,
) -> None:
    if feature_matrix.sample_keys != dataset.sample_keys:
        raise ValueError("Project feature matrix sample keys do not match BNCI epoch order")
    np.testing.assert_array_equal(feature_matrix.y, dataset.y)


def _build_feature_logreg_pipeline(
    benchmark_config: BNCIProjectFeatureBenchmarkConfig,
) -> Pipeline:
    logistic = LogisticRegression(
        C=benchmark_config.logistic_c,
        solver=benchmark_config.logistic_solver,
        class_weight=benchmark_config.logistic_class_weight,
        max_iter=benchmark_config.logistic_max_iter,
    )
    classifier = (
        OneVsRestClassifier(logistic)
        if benchmark_config.logistic_solver == "liblinear"
        else logistic
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", classifier),
        ]
    )
