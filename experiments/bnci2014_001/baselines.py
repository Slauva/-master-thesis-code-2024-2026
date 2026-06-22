from dataclasses import dataclass
from typing import Any

import mne
import numpy as np
from mne.decoding import CSP
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from experiments.bnci2014_001.config import BNCICSPBaselineConfig, BNCISplitConfig
from experiments.bnci2014_001.data import BNCIEpochDataset, BNCISplit, audit_split

CSP_LDA_BASELINE_VERSION = 1


@dataclass(frozen=True, slots=True)
class CSPBaselinePrediction:
    split_name: str
    y_true: NDArray[np.int64]
    y_pred: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    probabilities: NDArray[np.float64] | None


def fit_predict_csp_lda(
    dataset: BNCIEpochDataset,
    split: BNCISplit,
    *,
    baseline_config: BNCICSPBaselineConfig,
    split_config: BNCISplitConfig,
) -> CSPBaselinePrediction:
    validate_training_split(dataset, split, split_config=split_config)

    train_indices = np.asarray(split.train_indices, dtype=np.int64)
    test_indices = np.asarray(split.test_indices, dtype=np.int64)
    X_train = np.asarray(dataset.X[train_indices], dtype=np.float64).copy()
    y_train = np.asarray(dataset.y[train_indices], dtype=np.int64).copy()
    X_test = np.asarray(dataset.X[test_indices], dtype=np.float64).copy()
    y_test = np.asarray(dataset.y[test_indices], dtype=np.int64).copy()

    csp = CSP(
        n_components=baseline_config.n_components,
        reg=baseline_config.reg,
        log=baseline_config.log,
        cov_est=baseline_config.cov_est,
        norm_trace=baseline_config.norm_trace,
        component_order=baseline_config.component_order,
    )
    lda_kwargs: dict[str, Any] = {"solver": baseline_config.lda_solver}
    if baseline_config.lda_shrinkage is not None:
        lda_kwargs["shrinkage"] = baseline_config.lda_shrinkage
    lda = LinearDiscriminantAnalysis(**lda_kwargs)

    with mne.use_log_level("WARNING"):
        X_train_csp = csp.fit_transform(X_train, y_train)
        lda.fit(X_train_csp, y_train)
        X_test_csp = csp.transform(X_test)
    y_pred = np.asarray(lda.predict(X_test_csp), dtype=np.int64)
    probabilities = _predict_probabilities(lda, X_test_csp)
    y_test.setflags(write=False)
    y_pred.setflags(write=False)
    test_indices.setflags(write=False)
    if probabilities is not None:
        probabilities.setflags(write=False)
    return CSPBaselinePrediction(
        split_name=split.name,
        y_true=y_test,
        y_pred=y_pred,
        test_indices=test_indices,
        probabilities=probabilities,
    )


def validate_training_split(
    dataset: BNCIEpochDataset,
    split: BNCISplit,
    *,
    split_config: BNCISplitConfig,
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


def _predict_probabilities(
    lda: LinearDiscriminantAnalysis,
    X_test_csp: NDArray[np.floating[Any]],
) -> NDArray[np.float64] | None:
    if not hasattr(lda, "predict_proba"):
        return None
    probabilities = np.asarray(lda.predict_proba(X_test_csp), dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("LDA probabilities must be two-dimensional")
    return probabilities
