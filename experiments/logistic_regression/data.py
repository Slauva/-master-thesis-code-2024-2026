import hashlib
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from experiments.logistic_regression.config import SubjectSplitConfig
from experiments.logistic_regression.schemas import LeakageAudit, PixelTargetDataset, SubjectSplit
from utils.datasets.schemas import RandomSample, Sample


def build_random_imagery_targets(
    samples: Sequence[Sample],
    *,
    image_rows: int = 6,
    image_columns: int = 6,
) -> PixelTargetDataset:
    if not samples:
        raise ValueError("At least one random imagery sample is required")
    if image_rows < 1 or image_columns < 1:
        raise ValueError("Image dimensions must be positive")

    ordered = sorted(samples, key=lambda sample: (sample.subject_id, sample.trial_number, sample.block_index))
    images: list[np.ndarray] = []
    sample_keys: list[tuple[int, int, int]] = []
    seeds: list[int] = []
    fingerprints: list[str] = []
    for sample in ordered:
        if not isinstance(sample, RandomSample):
            raise TypeError("Pixel reconstruction targets require only RandomSample records")
        image = np.asarray(sample.img)
        if image.shape != (image_rows, image_columns):
            raise ValueError(
                f"Random image for subject={sample.subject_id}, trial={sample.trial_number}, "
                f"block={sample.block_index} has shape {image.shape}, expected "
                f"({image_rows}, {image_columns})"
            )
        if not np.isin(image, (0, 1)).all():
            raise ValueError("Random imagery targets must contain only zero and one")
        image = np.asarray(image, dtype=np.int8)
        images.append(image.reshape(-1))
        sample_keys.append((sample.subject_id, sample.trial_number, sample.block_index))
        seeds.append(sample.seed)
        fingerprints.append(hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())

    y = np.stack(images).astype(np.int8, copy=False)
    subject_ids = np.asarray([key[0] for key in sample_keys], dtype=np.int64)
    trial_numbers = np.asarray([key[1] for key in sample_keys], dtype=np.int64)
    block_indices = np.asarray([key[2] for key in sample_keys], dtype=np.int64)
    seed_array = np.asarray(seeds, dtype=np.int64)
    for array in (y, subject_ids, trial_numbers, block_indices, seed_array):
        array.setflags(write=False)

    pixel_names = tuple(
        f"pixel_r{row}_c{column}"
        for row in range(image_rows)
        for column in range(image_columns)
    )
    return PixelTargetDataset(
        y=y,
        pixel_names=pixel_names,
        sample_keys=tuple(sample_keys),
        subject_ids=subject_ids,
        trial_numbers=trial_numbers,
        block_indices=block_indices,
        seeds=seed_array,
        image_fingerprints=tuple(fingerprints),
    )


def create_subject_split(
    targets: PixelTargetDataset,
    *,
    config: SubjectSplitConfig,
) -> tuple[SubjectSplit, LeakageAudit]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    rows = np.arange(targets.y.shape[0], dtype=np.int64)
    train_indices, test_indices = next(
        splitter.split(rows[:, np.newaxis], groups=targets.subject_ids)
    )
    train_indices = np.sort(train_indices.astype(np.int64, copy=False))
    test_indices = np.sort(test_indices.astype(np.int64, copy=False))
    split = SubjectSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        train_subjects=tuple(int(value) for value in np.unique(targets.subject_ids[train_indices])),
        test_subjects=tuple(int(value) for value in np.unique(targets.subject_ids[test_indices])),
        n_samples=targets.y.shape[0],
        random_state=config.random_state,
        test_size=config.test_size,
    )
    audit = audit_subject_split(targets, split)
    if audit.has_leakage:
        raise ValueError("Subject split leaks subjects, sample keys, seeds, or image payloads")
    if config.require_both_classes and not audit.all_tasks_have_both_classes:
        raise ValueError("Every pixel task must contain both classes in train and test")
    return split, audit


def audit_subject_split(
    targets: PixelTargetDataset,
    split: SubjectSplit,
) -> LeakageAudit:
    train = split.train_indices
    test = split.test_indices
    train_keys = {targets.sample_keys[int(index)] for index in train}
    test_keys = {targets.sample_keys[int(index)] for index in test}
    train_fingerprints = {targets.image_fingerprints[int(index)] for index in train}
    test_fingerprints = {targets.image_fingerprints[int(index)] for index in test}
    train_seeds = {int(value) for value in targets.seeds[train]}
    test_seeds = {int(value) for value in targets.seeds[test]}

    train_positive_counts = targets.y[train].sum(axis=0, dtype=np.int64)
    test_positive_counts = targets.y[test].sum(axis=0, dtype=np.int64)
    train_has_both = (train_positive_counts > 0) & (train_positive_counts < train.size)
    test_has_both = (test_positive_counts > 0) & (test_positive_counts < test.size)
    return LeakageAudit(
        overlapping_subjects=tuple(sorted(set(split.train_subjects) & set(split.test_subjects))),
        overlapping_sample_keys=tuple(sorted(train_keys & test_keys)),
        overlapping_seeds=tuple(sorted(train_seeds & test_seeds)),
        overlapping_image_fingerprints=tuple(sorted(train_fingerprints & test_fingerprints)),
        train_positive_counts=train_positive_counts,
        test_positive_counts=test_positive_counts,
        all_tasks_have_both_classes=bool(np.all(train_has_both & test_has_both)),
    )
