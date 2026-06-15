from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    per_pixel_balanced_accuracy: NDArray[np.float64]
    per_pixel_macro_f1: NDArray[np.float64]
    per_pixel_brier_score: NDArray[np.float64]
    per_sample_iou: NDArray[np.float64]
    mean_balanced_accuracy: float
    mean_macro_f1: float
    mean_brier_score: float
    mean_sample_iou: float
    micro_iou: float
    bit_accuracy: float
    exact_match_accuracy: float
    mean_hamming_distance: float
    hamming_loss: float

    def __post_init__(self) -> None:
        n_pixels = self.per_pixel_balanced_accuracy.size
        if n_pixels < 1:
            raise ValueError("Prediction metrics require at least one pixel")
        for name, values in (
            ("per_pixel_balanced_accuracy", self.per_pixel_balanced_accuracy),
            ("per_pixel_macro_f1", self.per_pixel_macro_f1),
            ("per_pixel_brier_score", self.per_pixel_brier_score),
        ):
            if values.shape != (n_pixels,) or values.dtype != np.dtype(np.float64):
                raise TypeError(f"`{name}` must be a float64 vector matching the pixel axis")
            if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"`{name}` must contain finite values in [0, 1]")

        if self.per_sample_iou.ndim != 1 or self.per_sample_iou.size < 1:
            raise TypeError("`per_sample_iou` must be a non-empty vector")
        if self.per_sample_iou.dtype != np.dtype(np.float64):
            raise TypeError("`per_sample_iou` must be a float64 vector")
        if not np.isfinite(self.per_sample_iou).all() or np.any(
            (self.per_sample_iou < 0.0) | (self.per_sample_iou > 1.0)
        ):
            raise ValueError("`per_sample_iou` must contain finite values in [0, 1]")

        for name, value in (
            ("mean_balanced_accuracy", self.mean_balanced_accuracy),
            ("mean_macro_f1", self.mean_macro_f1),
            ("mean_brier_score", self.mean_brier_score),
            ("mean_sample_iou", self.mean_sample_iou),
            ("micro_iou", self.micro_iou),
            ("bit_accuracy", self.bit_accuracy),
            ("exact_match_accuracy", self.exact_match_accuracy),
            ("hamming_loss", self.hamming_loss),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"`{name}` must be finite and in [0, 1]")
        if not np.isfinite(self.mean_hamming_distance) or not 0.0 <= self.mean_hamming_distance <= n_pixels:
            raise ValueError("Mean Hamming distance must be finite and match the pixel count")
        aggregate_pairs = (
            (
                "mean_balanced_accuracy",
                self.mean_balanced_accuracy,
                self.per_pixel_balanced_accuracy.mean(),
            ),
            ("mean_macro_f1", self.mean_macro_f1, self.per_pixel_macro_f1.mean()),
            (
                "mean_brier_score",
                self.mean_brier_score,
                self.per_pixel_brier_score.mean(),
            ),
            ("mean_sample_iou", self.mean_sample_iou, self.per_sample_iou.mean()),
            ("hamming_loss", self.hamming_loss, 1.0 - self.bit_accuracy),
            (
                "mean_hamming_distance",
                self.mean_hamming_distance,
                self.hamming_loss * n_pixels,
            ),
        )
        for name, value, expected in aggregate_pairs:
            if not np.isclose(value, expected, rtol=0.0, atol=1e-12):
                raise ValueError(f"`{name}` is inconsistent with its component metrics")


@dataclass(frozen=True, slots=True)
class SubjectBootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    n_attempts: int
    samples: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("Bootstrap confidence level must be between zero and one")
        if self.n_resamples < 1 or self.n_attempts < self.n_resamples:
            raise ValueError("Bootstrap counts must describe completed resampling")
        if self.samples.shape != (self.n_resamples,) or self.samples.dtype != np.dtype(np.float64):
            raise TypeError("Bootstrap samples must be a float64 vector matching `n_resamples`")
        if not np.isfinite(self.samples).all() or np.any((self.samples < 0.0) | (self.samples > 1.0)):
            raise ValueError("Bootstrap samples must be finite and in [0, 1]")
        if not 0.0 <= self.lower <= self.upper <= 1.0:
            raise ValueError("Bootstrap interval must be ordered and contained in [0, 1]")
        if not np.isfinite(self.estimate) or not 0.0 <= self.estimate <= 1.0:
            raise ValueError("Bootstrap estimate must be finite and in [0, 1]")


def evaluate_prediction_matrix(
    y_true: ArrayLike,
    predictions: ArrayLike,
    probabilities: ArrayLike,
) -> PredictionMetrics:
    targets, labels = _validate_binary_matrices(y_true, predictions)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.shape != targets.shape:
        raise ValueError("Probabilities must match target and prediction shape")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Probabilities must be finite and in [0, 1]")

    balanced_accuracy = _per_pixel_balanced_accuracy(targets, labels)
    macro_f1 = _per_pixel_macro_f1(targets, labels)
    brier_score = np.mean((scores - targets) ** 2, axis=0, dtype=np.float64)
    mismatches = np.count_nonzero(targets != labels, axis=1)
    intersection = np.count_nonzero((targets == 1) & (labels == 1), axis=1)
    union = np.count_nonzero((targets == 1) | (labels == 1), axis=1)
    sample_iou = np.divide(
        intersection,
        union,
        out=np.ones(targets.shape[0], dtype=np.float64),
        where=union != 0,
    )
    total_union = int(union.sum())
    micro_iou = 1.0 if total_union == 0 else float(intersection.sum() / total_union)
    hamming_loss = float(np.mean(targets != labels))

    for values in (balanced_accuracy, macro_f1, brier_score, sample_iou):
        values.setflags(write=False)
    return PredictionMetrics(
        per_pixel_balanced_accuracy=balanced_accuracy,
        per_pixel_macro_f1=macro_f1,
        per_pixel_brier_score=brier_score,
        per_sample_iou=sample_iou,
        mean_balanced_accuracy=float(balanced_accuracy.mean()),
        mean_macro_f1=float(macro_f1.mean()),
        mean_brier_score=float(brier_score.mean()),
        mean_sample_iou=float(sample_iou.mean()),
        micro_iou=micro_iou,
        bit_accuracy=float(np.mean(targets == labels)),
        exact_match_accuracy=float(np.mean(mismatches == 0)),
        mean_hamming_distance=float(mismatches.mean()),
        hamming_loss=hamming_loss,
    )


def bootstrap_subject_mean_balanced_accuracy(
    y_true: ArrayLike,
    predictions: ArrayLike,
    subject_ids: ArrayLike,
    *,
    n_resamples: int,
    random_state: int,
    confidence_level: float = 0.95,
) -> SubjectBootstrapInterval:
    targets, labels = _validate_binary_matrices(y_true, predictions)
    groups = np.asarray(subject_ids)
    if groups.ndim != 1 or groups.shape[0] != targets.shape[0]:
        raise ValueError("Subject IDs must be a vector matching target rows")
    unique_subjects = np.unique(groups)
    if unique_subjects.size < 2:
        raise ValueError("Subject bootstrap requires at least two distinct subjects")
    if isinstance(n_resamples, bool) or n_resamples < 1:
        raise ValueError("`n_resamples` must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("`confidence_level` must be between zero and one")

    subject_rows = {
        subject: np.flatnonzero(groups == subject)
        for subject in unique_subjects
    }
    rng = np.random.default_rng(random_state)
    bootstrap_scores = np.empty(n_resamples, dtype=np.float64)
    completed = 0
    attempts = 0
    max_attempts = n_resamples * 100
    while completed < n_resamples and attempts < max_attempts:
        attempts += 1
        drawn_subjects = rng.choice(
            unique_subjects,
            size=unique_subjects.size,
            replace=True,
        )
        row_indices = np.concatenate(
            [subject_rows[subject] for subject in drawn_subjects]
        )
        sampled_targets = targets[row_indices]
        if not _all_pixels_have_both_classes(sampled_targets):
            continue
        bootstrap_scores[completed] = float(
            _per_pixel_balanced_accuracy(
                sampled_targets,
                labels[row_indices],
            ).mean()
        )
        completed += 1
    if completed != n_resamples:
        raise RuntimeError(
            "Could not draw enough class-complete subject bootstrap samples "
            f"after {attempts} attempts"
        )

    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        bootstrap_scores,
        (alpha, 1.0 - alpha),
        method="linear",
    )
    bootstrap_scores.setflags(write=False)
    return SubjectBootstrapInterval(
        estimate=float(_per_pixel_balanced_accuracy(targets, labels).mean()),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        n_attempts=attempts,
        samples=bootstrap_scores,
    )


def _validate_binary_matrices(
    y_true: ArrayLike,
    predictions: ArrayLike,
) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
    targets = np.asarray(y_true)
    labels = np.asarray(predictions)
    if targets.ndim != 2 or min(targets.shape) < 1:
        raise ValueError("Targets must have non-empty shape (sample, pixel)")
    if labels.shape != targets.shape:
        raise ValueError("Predictions must match target shape")
    if not np.isin(targets, (0, 1)).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("Targets and predictions must be binary")
    targets = targets.astype(np.int8, copy=False)
    labels = labels.astype(np.int8, copy=False)
    if not _all_pixels_have_both_classes(targets):
        raise ValueError("Every evaluated pixel must contain both target classes")
    return targets, labels


def _all_pixels_have_both_classes(targets: NDArray[np.int8]) -> bool:
    positive_counts = targets.sum(axis=0, dtype=np.int64)
    return bool(
        np.all(positive_counts > 0)
        and np.all(positive_counts < targets.shape[0])
    )


def _per_pixel_balanced_accuracy(
    targets: NDArray[np.int8],
    predictions: NDArray[np.int8],
) -> NDArray[np.float64]:
    positive = targets == 1
    negative = ~positive
    sensitivity = np.sum((predictions == 1) & positive, axis=0) / np.sum(
        positive,
        axis=0,
    )
    specificity = np.sum((predictions == 0) & negative, axis=0) / np.sum(
        negative,
        axis=0,
    )
    return np.asarray((sensitivity + specificity) / 2.0, dtype=np.float64)


def _per_pixel_macro_f1(
    targets: NDArray[np.int8],
    predictions: NDArray[np.int8],
) -> NDArray[np.float64]:
    true_positive = np.sum((targets == 1) & (predictions == 1), axis=0)
    true_negative = np.sum((targets == 0) & (predictions == 0), axis=0)
    false_positive = np.sum((targets == 0) & (predictions == 1), axis=0)
    false_negative = np.sum((targets == 1) & (predictions == 0), axis=0)
    positive_f1 = (2.0 * true_positive) / (
        2.0 * true_positive + false_positive + false_negative
    )
    negative_f1 = (2.0 * true_negative) / (
        2.0 * true_negative + false_positive + false_negative
    )
    return np.asarray((positive_f1 + negative_f1) / 2.0, dtype=np.float64)
