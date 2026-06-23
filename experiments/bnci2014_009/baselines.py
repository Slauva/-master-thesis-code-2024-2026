from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.bnci2014_009.config import BNCI009ClassicalBenchmarkConfig, BNCI009SplitConfig
from experiments.bnci2014_009.data import BNCI009EpochDataset, BNCI009Split, audit_split
from experiments.bnci2014_009.features import (
    BNCI009ERPFeatureMatrix,
    fit_transform_xdawn_tangent_space,
)

CLASSICAL_BASELINE_VERSION = 3


@dataclass(frozen=True, slots=True)
class ClassicalFoldPrediction:
    model_id: str
    split_name: str
    y_true: NDArray[np.int64]
    y_pred: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    target_score: NDArray[np.float64] | None


def fit_predict_classical_variant(
    model_id: str,
    dataset: BNCI009EpochDataset,
    erp_features: BNCI009ERPFeatureMatrix,
    split: BNCI009Split,
    *,
    classical_config: BNCI009ClassicalBenchmarkConfig,
    split_config: BNCI009SplitConfig,
) -> ClassicalFoldPrediction:
    validate_training_split(dataset, split, split_config=split_config)
    validate_erp_alignment(dataset, erp_features)

    train_indices = np.asarray(split.train_indices, dtype=np.int64)
    test_indices = np.asarray(split.test_indices, dtype=np.int64)
    y_train = np.asarray(dataset.y[train_indices], dtype=np.int64)
    y_test = np.asarray(dataset.y[test_indices], dtype=np.int64)

    if model_id == "dummy-prior":
        classifier = DummyClassifier(strategy="prior")
        classifier.fit(np.zeros((train_indices.size, 1), dtype=np.float64), y_train)
        X_test_dummy = np.zeros((test_indices.size, 1), dtype=np.float64)
        y_pred = np.asarray(classifier.predict(X_test_dummy), dtype=np.int64)
        target_score = _target_score(classifier, X_test_dummy)
    elif model_id.startswith("erp-"):
        classifier = _build_erp_classifier(model_id, classical_config)
        X_train = np.asarray(erp_features.X[train_indices], dtype=np.float64)
        X_test = np.asarray(erp_features.X[test_indices], dtype=np.float64)
        classifier.fit(X_train, y_train)
        y_pred = np.asarray(classifier.predict(X_test), dtype=np.int64)
        target_score = _target_score(classifier, X_test)
    elif model_id.startswith("xdawn-tangent-"):
        transformed = fit_transform_xdawn_tangent_space(
            dataset.X[train_indices],
            y_train,
            dataset.X[test_indices],
            n_filters=classical_config.xdawn_n_filters,
            estimator=classical_config.xdawn_estimator,
            tangent_metric=classical_config.tangent_metric,
        )
        classifier = _build_xdawn_classifier(model_id, classical_config)
        classifier.fit(transformed.X_train, y_train)
        y_pred = np.asarray(classifier.predict(transformed.X_apply), dtype=np.int64)
        target_score = _target_score(classifier, transformed.X_apply)
    else:
        raise ValueError(f"Unsupported BNCI2014_009 classical model id: {model_id!r}")

    y_test.setflags(write=False)
    y_pred.setflags(write=False)
    test_indices.setflags(write=False)
    if target_score is not None:
        target_score.setflags(write=False)
    return ClassicalFoldPrediction(
        model_id=model_id,
        split_name=split.name,
        y_true=y_test,
        y_pred=y_pred,
        test_indices=test_indices,
        target_score=target_score,
    )


def validate_training_split(
    dataset: BNCI009EpochDataset,
    split: BNCI009Split,
    *,
    split_config: BNCI009SplitConfig,
) -> None:
    if split.n_samples != dataset.y.shape[0]:
        raise ValueError("Split sample count does not match the dataset")
    audit = audit_split(dataset, split)
    if audit.has_forbidden_leakage:
        raise ValueError(f"Split {split.name!r} has forbidden leakage")
    if split_config.require_all_classes_in_train and not audit.all_train_classes_present:
        raise ValueError(f"Split {split.name!r} is missing at least one train class")
    if split_config.require_all_classes_in_test and not audit.all_test_classes_present:
        raise ValueError(f"Split {split.name!r} is missing at least one test class")


def validate_erp_alignment(
    dataset: BNCI009EpochDataset,
    erp_features: BNCI009ERPFeatureMatrix,
) -> None:
    if erp_features.sample_keys != dataset.sample_keys:
        raise ValueError("ERP feature matrix sample keys do not match BNCI2014_009 epoch order")
    np.testing.assert_array_equal(erp_features.y, dataset.y)


def _build_erp_classifier(
    model_id: str,
    config: BNCI009ClassicalBenchmarkConfig,
) -> Pipeline:
    if model_id == "erp-lda":
        estimator: Any = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif model_id == "erp-logreg":
        estimator = LogisticRegression(
            C=config.logistic_c,
            solver="liblinear",
            class_weight="balanced",
            max_iter=config.logistic_max_iter,
        )
    elif model_id == "erp-linear-svm":
        estimator = SGDClassifier(
            loss="hinge",
            alpha=config.svm_alpha,
            class_weight="balanced",
            max_iter=config.svm_max_iter,
            tol=config.svm_tol,
            random_state=config.seed,
        )
    elif model_id == "erp-ridge":
        estimator = RidgeClassifier(class_weight="balanced")
    else:
        raise ValueError(f"Unsupported ERP model id: {model_id!r}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def _build_xdawn_classifier(
    model_id: str,
    config: BNCI009ClassicalBenchmarkConfig,
) -> Pipeline:
    if model_id == "xdawn-tangent-lda":
        estimator: Any = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif model_id == "xdawn-tangent-logreg":
        estimator = LogisticRegression(
            C=config.logistic_c,
            solver="liblinear",
            class_weight="balanced",
            max_iter=config.logistic_max_iter,
        )
    else:
        raise ValueError(f"Unsupported xDAWN tangent model id: {model_id!r}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def _target_score(
    classifier: Any,
    X: NDArray[np.floating[Any]],
) -> NDArray[np.float64] | None:
    classes = np.asarray(getattr(classifier, "classes_", (0, 1)), dtype=np.int64)
    if hasattr(classifier, "predict_proba"):
        probabilities = np.asarray(classifier.predict_proba(X), dtype=np.float64)
        target_columns = np.flatnonzero(classes == 0)
        if probabilities.ndim == 2 and target_columns.size == 1:
            return np.asarray(probabilities[:, int(target_columns[0])], dtype=np.float64)
    if hasattr(classifier, "decision_function"):
        decision = np.asarray(classifier.decision_function(X), dtype=np.float64)
        if decision.ndim == 1:
            if classes.shape == (2,) and int(classes[1]) == 0:
                return decision
            return -decision
        target_columns = np.flatnonzero(classes == 0)
        if decision.ndim == 2 and target_columns.size == 1:
            return np.asarray(decision[:, int(target_columns[0])], dtype=np.float64)
    return None
