from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from numpy.typing import NDArray

from experiments.bnci2014_001.config import BNCI_LABELS, BNCI2014001Config

BNCISampleKey = tuple[int, str, str, int]


@dataclass(frozen=True, slots=True)
class BNCIEpochMetadata:
    subject: int
    session: str
    run: str
    epoch_index: int
    label: str
    y: int

    @property
    def sample_key(self) -> BNCISampleKey:
        return (self.subject, self.session, self.run, self.epoch_index)


@dataclass(frozen=True, slots=True)
class BNCIEpochDataset:
    X: NDArray[np.floating[Any]]
    y: NDArray[np.integer[Any]]
    metadata: tuple[BNCIEpochMetadata, ...]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.X.ndim != 3:
            raise ValueError("X must have shape (epoch, channel, time)")
        if self.y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if self.X.shape[0] != self.y.shape[0] or self.X.shape[0] != len(self.metadata):
            raise ValueError("X, y, and metadata must contain the same number of epochs")
        if not np.issubdtype(self.X.dtype, np.floating):
            raise TypeError("X must have a floating-point dtype")
        if not np.issubdtype(self.y.dtype, np.integer):
            raise TypeError("y must have an integer dtype")
        if not np.isfinite(self.X).all():
            raise ValueError("X must contain only finite values")
        if len(set(self.sample_keys)) != len(self.sample_keys):
            raise ValueError("Epoch sample keys must be unique")
        if tuple(self.class_names) != tuple(dict.fromkeys(self.class_names)):
            raise ValueError("Class names must be unique")
        observed_y = {int(value) for value in np.unique(self.y)}
        expected_y = set(range(len(self.class_names)))
        if not observed_y <= expected_y:
            raise ValueError(f"Class indices {sorted(observed_y)} exceed class names {self.class_names}")

    @property
    def sample_keys(self) -> tuple[BNCISampleKey, ...]:
        return tuple(row.sample_key for row in self.metadata)

    @property
    def subjects(self) -> NDArray[np.int64]:
        values = np.asarray([row.subject for row in self.metadata], dtype=np.int64)
        values.setflags(write=False)
        return values

    @property
    def sessions(self) -> tuple[str, ...]:
        return tuple(row.session for row in self.metadata)

    @property
    def runs(self) -> tuple[str, ...]:
        return tuple(row.run for row in self.metadata)


@dataclass(frozen=True, slots=True)
class BNCISplit:
    name: str
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    train_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]
    n_samples: int

    def __post_init__(self) -> None:
        if self.train_indices.ndim != 1 or self.test_indices.ndim != 1:
            raise ValueError("Split indices must be one-dimensional")
        if self.train_indices.size == 0 or self.test_indices.size == 0:
            raise ValueError("Train and test indices must be non-empty")
        if self.n_samples < self.train_indices.size + self.test_indices.size:
            raise ValueError("n_samples is smaller than the split index count")
        if np.intersect1d(self.train_indices, self.test_indices).size:
            raise ValueError("Train and test indices must be disjoint")


@dataclass(frozen=True, slots=True)
class BNCISplitAudit:
    split_name: str
    overlapping_subjects: tuple[int, ...]
    overlapping_sample_keys: tuple[BNCISampleKey, ...]
    train_class_counts: dict[str, int]
    test_class_counts: dict[str, int]
    all_train_classes_present: bool
    all_test_classes_present: bool

    @property
    def has_subject_leakage(self) -> bool:
        return bool(self.overlapping_subjects)

    @property
    def has_sample_key_leakage(self) -> bool:
        return bool(self.overlapping_sample_keys)

    @property
    def has_forbidden_leakage(self) -> bool:
        return self.has_subject_leakage or self.has_sample_key_leakage


def build_epoch_dataset(
    X: NDArray[np.floating[Any]],
    labels: NDArray[Any] | list[str],
    metadata: pd.DataFrame,
    *,
    class_names: tuple[str, ...] = BNCI_LABELS,
    dtype: np.dtype[Any] | str = np.float32,
) -> BNCIEpochDataset:
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if X.shape[0] != labels_array.shape[0] or X.shape[0] != len(metadata):
        raise ValueError("X, labels, and metadata row counts must match")

    required_columns = {"subject", "session", "run"}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required column(s): {sorted(missing)}")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be unique")

    label_to_index = {label: index for index, label in enumerate(class_names)}
    unknown_labels = sorted({str(label) for label in labels_array if str(label) not in label_to_index})
    if unknown_labels:
        raise ValueError(f"Unknown BNCI labels: {unknown_labels}")

    frame = metadata.reset_index(drop=True).copy()
    frame["label"] = [str(label) for label in labels_array]
    frame["epoch_index"] = frame.groupby(["subject", "session", "run"], sort=False).cumcount()

    rows = tuple(
        BNCIEpochMetadata(
            subject=int(record.subject),
            session=str(record.session),
            run=str(record.run),
            epoch_index=int(record.epoch_index),
            label=str(record.label),
            y=label_to_index[str(record.label)],
        )
        for record in frame.itertuples(index=False)
    )
    y = np.asarray([row.y for row in rows], dtype=np.int64)
    y.setflags(write=False)
    X_array = np.asarray(X, dtype=dtype)
    X_array.setflags(write=False)
    return BNCIEpochDataset(
        X=X_array,
        y=y,
        metadata=rows,
        class_names=class_names,
    )


def load_bnci_epochs(
    config: BNCI2014001Config,
    *,
    subjects: tuple[int, ...] | None = None,
) -> BNCIEpochDataset:
    selected_subjects = subjects or config.dataset.subjects
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=config.dataset.n_classes)
    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=list(selected_subjects))
    return build_epoch_dataset(
        X,
        labels,
        metadata,
        class_names=config.dataset.labels,
        dtype=config.dataset.dtype,
    )


def create_leave_one_subject_splits(dataset: BNCIEpochDataset) -> tuple[BNCISplit, ...]:
    subjects = tuple(int(value) for value in np.unique(dataset.subjects))
    if len(subjects) < 2:
        raise ValueError("Leave-one-subject-out protocol requires at least two subjects")

    splits: list[BNCISplit] = []
    for subject in subjects:
        test_indices = np.flatnonzero(dataset.subjects == subject).astype(np.int64, copy=False)
        train_indices = np.flatnonzero(dataset.subjects != subject).astype(np.int64, copy=False)
        train_indices.setflags(write=False)
        test_indices.setflags(write=False)
        train_subjects = tuple(value for value in subjects if value != subject)
        splits.append(
            BNCISplit(
                name=f"leave-subject-{subject}-out",
                train_indices=train_indices,
                test_indices=test_indices,
                train_subjects=train_subjects,
                test_subjects=(subject,),
                n_samples=dataset.y.shape[0],
            )
        )
    return tuple(splits)


def audit_split(dataset: BNCIEpochDataset, split: BNCISplit) -> BNCISplitAudit:
    train_subjects = set(split.train_subjects)
    test_subjects = set(split.test_subjects)
    train_keys = {dataset.sample_keys[int(index)] for index in split.train_indices}
    test_keys = {dataset.sample_keys[int(index)] for index in split.test_indices}
    train_labels = [dataset.class_names[int(dataset.y[int(index)])] for index in split.train_indices]
    test_labels = [dataset.class_names[int(dataset.y[int(index)])] for index in split.test_indices]
    train_counts = _complete_counts(train_labels, dataset.class_names)
    test_counts = _complete_counts(test_labels, dataset.class_names)
    return BNCISplitAudit(
        split_name=split.name,
        overlapping_subjects=tuple(sorted(train_subjects & test_subjects)),
        overlapping_sample_keys=tuple(sorted(train_keys & test_keys)),
        train_class_counts=train_counts,
        test_class_counts=test_counts,
        all_train_classes_present=all(count > 0 for count in train_counts.values()),
        all_test_classes_present=all(count > 0 for count in test_counts.values()),
    )


def _complete_counts(labels: list[str], class_names: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(labels)
    return {label: int(counts.get(label, 0)) for label in class_names}
