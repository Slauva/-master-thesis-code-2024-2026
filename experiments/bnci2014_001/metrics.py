from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class_recall: dict[str, float]
    confusion_matrix: NDArray[np.int64]
    n_samples: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "per_class_recall": self.per_class_recall,
            "confusion_matrix": self.confusion_matrix.astype(int).tolist(),
            "n_samples": self.n_samples,
        }


def evaluate_multiclass_predictions(
    y_true: NDArray[np.integer[Any]],
    y_pred: NDArray[np.integer[Any]],
    *,
    class_names: tuple[str, ...],
) -> ClassificationMetrics:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("At least one prediction is required")
    if not class_names:
        raise ValueError("class_names must not be empty")

    labels = np.arange(len(class_names), dtype=np.int64)
    for name, values in (("y_true", true), ("y_pred", pred)):
        observed = set(int(value) for value in np.unique(values))
        expected = set(int(value) for value in labels)
        if not observed <= expected:
            raise ValueError(f"{name} contains class indices outside {class_names}: {sorted(observed - expected)}")

    recalls = recall_score(
        true,
        pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return ClassificationMetrics(
        accuracy=float(accuracy_score(true, pred)),
        balanced_accuracy=float(balanced_accuracy_score(true, pred)),
        macro_f1=float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        per_class_recall={
            class_name: float(recall)
            for class_name, recall in zip(class_names, recalls, strict=True)
        },
        confusion_matrix=confusion_matrix(true, pred, labels=labels).astype(np.int64, copy=False),
        n_samples=int(true.size),
    )


def summarize_fold_metrics(fold_metrics: tuple[ClassificationMetrics, ...]) -> dict[str, Any]:
    if not fold_metrics:
        raise ValueError("At least one fold metric payload is required")

    accuracy = np.asarray([fold.accuracy for fold in fold_metrics], dtype=np.float64)
    balanced_accuracy = np.asarray([fold.balanced_accuracy for fold in fold_metrics], dtype=np.float64)
    macro_f1 = np.asarray([fold.macro_f1 for fold in fold_metrics], dtype=np.float64)
    return {
        "n_folds": len(fold_metrics),
        "n_samples": int(sum(fold.n_samples for fold in fold_metrics)),
        "accuracy_mean": float(np.mean(accuracy)),
        "accuracy_std": float(np.std(accuracy, ddof=0)),
        "balanced_accuracy_mean": float(np.mean(balanced_accuracy)),
        "balanced_accuracy_std": float(np.std(balanced_accuracy, ddof=0)),
        "macro_f1_mean": float(np.mean(macro_f1)),
        "macro_f1_std": float(np.std(macro_f1, ddof=0)),
        "confusion_matrix": np.sum(
            [fold.confusion_matrix for fold in fold_metrics],
            axis=0,
            dtype=np.int64,
        ).tolist(),
    }
