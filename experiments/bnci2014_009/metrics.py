from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class BinaryP300Metrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    target_recall: float
    non_target_recall: float
    roc_auc: float | None
    pr_auc: float | None
    confusion_matrix: NDArray[np.int64]
    n_samples: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "target_recall": self.target_recall,
            "non_target_recall": self.non_target_recall,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "confusion_matrix": self.confusion_matrix.astype(int).tolist(),
            "n_samples": self.n_samples,
        }


def evaluate_binary_p300_predictions(
    y_true: NDArray[np.integer[Any]],
    y_pred: NDArray[np.integer[Any]],
    *,
    target_score: NDArray[np.floating[Any]] | None = None,
    class_names: tuple[str, ...],
) -> BinaryP300Metrics:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("At least one prediction is required")
    if class_names != ("Target", "NonTarget"):
        raise ValueError("BNCI2014_009 metrics require class order ('Target', 'NonTarget')")
    observed = set(int(value) for value in np.unique(np.concatenate([true, pred])))
    if not observed <= {0, 1}:
        raise ValueError(f"Binary P300 predictions contain unknown class indices: {sorted(observed - {0, 1})}")

    recalls = recall_score(true, pred, labels=np.asarray([0, 1]), average=None, zero_division=0)
    roc_auc: float | None = None
    pr_auc: float | None = None
    if target_score is not None:
        score = np.asarray(target_score, dtype=np.float64)
        if score.ndim != 1 or score.shape != true.shape:
            raise ValueError("target_score must be one-dimensional and align with y_true")
        if not np.isfinite(score).all():
            raise ValueError("target_score must contain only finite values")
        target_true = (true == 0).astype(np.int64)
        if len(set(int(value) for value in np.unique(target_true))) == 2:
            roc_auc = float(roc_auc_score(target_true, score))
            pr_auc = float(average_precision_score(target_true, score))

    return BinaryP300Metrics(
        accuracy=float(accuracy_score(true, pred)),
        balanced_accuracy=float(balanced_accuracy_score(true, pred)),
        macro_f1=float(f1_score(true, pred, labels=[0, 1], average="macro", zero_division=0)),
        target_recall=float(recalls[0]),
        non_target_recall=float(recalls[1]),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=confusion_matrix(true, pred, labels=[0, 1]).astype(np.int64, copy=False),
        n_samples=int(true.size),
    )


def summarize_fold_metrics(fold_metrics: tuple[BinaryP300Metrics, ...]) -> dict[str, Any]:
    if not fold_metrics:
        raise ValueError("At least one fold metric payload is required")

    accuracy = np.asarray([fold.accuracy for fold in fold_metrics], dtype=np.float64)
    balanced_accuracy = np.asarray([fold.balanced_accuracy for fold in fold_metrics], dtype=np.float64)
    macro_f1 = np.asarray([fold.macro_f1 for fold in fold_metrics], dtype=np.float64)
    target_recall = np.asarray([fold.target_recall for fold in fold_metrics], dtype=np.float64)
    non_target_recall = np.asarray([fold.non_target_recall for fold in fold_metrics], dtype=np.float64)
    roc_auc_values = [fold.roc_auc for fold in fold_metrics if fold.roc_auc is not None]
    pr_auc_values = [fold.pr_auc for fold in fold_metrics if fold.pr_auc is not None]
    return {
        "n_folds": len(fold_metrics),
        "n_samples": int(sum(fold.n_samples for fold in fold_metrics)),
        "accuracy_mean": float(np.mean(accuracy)),
        "accuracy_std": float(np.std(accuracy, ddof=0)),
        "balanced_accuracy_mean": float(np.mean(balanced_accuracy)),
        "balanced_accuracy_std": float(np.std(balanced_accuracy, ddof=0)),
        "macro_f1_mean": float(np.mean(macro_f1)),
        "macro_f1_std": float(np.std(macro_f1, ddof=0)),
        "target_recall_mean": float(np.mean(target_recall)),
        "target_recall_std": float(np.std(target_recall, ddof=0)),
        "non_target_recall_mean": float(np.mean(non_target_recall)),
        "non_target_recall_std": float(np.std(non_target_recall, ddof=0)),
        "roc_auc_mean": float(np.mean(roc_auc_values)) if roc_auc_values else None,
        "roc_auc_std": float(np.std(roc_auc_values, ddof=0)) if roc_auc_values else None,
        "pr_auc_mean": float(np.mean(pr_auc_values)) if pr_auc_values else None,
        "pr_auc_std": float(np.std(pr_auc_values, ddof=0)) if pr_auc_values else None,
        "confusion_matrix": np.sum(
            [fold.confusion_matrix for fold in fold_metrics],
            axis=0,
            dtype=np.int64,
        ).tolist(),
    }
