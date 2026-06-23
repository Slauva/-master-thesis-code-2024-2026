"""Shared random-imagery target and evaluation-protocol construction."""

import hashlib
from collections.abc import Sequence
from typing import Literal

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from experiments.logistic_regression.schemas import (
    EvaluationDirection,
    EvaluationProtocol,
    EvaluationProtocolDefinition,
    LeakageAudit,
    PixelTargetDataset,
    ProtocolLeakageAudit,
    SubjectSplit,
)
from experiments.random_imagery.config import SubjectSplitConfig
from utils.datasets.schemas import GeometricSample, RandomSample, Sample

TargetSampleType = Literal["geometric", "random"]
_NO_RANDOM_SEED = -1


def build_random_imagery_targets(
    samples: Sequence[Sample],
    *,
    image_rows: int = 6,
    image_columns: int = 6,
    allowed_sample_types: tuple[TargetSampleType, ...] = ("random",),
) -> PixelTargetDataset:
    if not samples:
        raise ValueError("At least one imagery sample is required")
    if image_rows < 1 or image_columns < 1:
        raise ValueError("Image dimensions must be positive")
    if not allowed_sample_types:
        raise ValueError("At least one target sample type must be allowed")
    if len(set(allowed_sample_types)) != len(allowed_sample_types):
        raise ValueError("Allowed target sample types must be unique")

    ordered = sorted(samples, key=lambda sample: (sample.subject_id, sample.trial_number, sample.block_index))
    images: list[np.ndarray] = []
    sample_keys: list[tuple[int, int, int]] = []
    seeds: list[int] = []
    sample_types: list[str] = []
    pattern_ids: list[int] = []
    fingerprints: list[str] = []
    for sample in ordered:
        if sample.type not in allowed_sample_types:
            if allowed_sample_types == ("random",):
                raise TypeError("Pixel reconstruction targets require only RandomSample records")
            allowed = ", ".join(allowed_sample_types)
            raise TypeError(
                f"Pixel reconstruction target sample type {sample.type!r} is not allowed; "
                f"expected one of: {allowed}"
            )
        image = np.asarray(sample.img)
        if image.shape != (image_rows, image_columns):
            raise ValueError(
                f"Image for subject={sample.subject_id}, trial={sample.trial_number}, "
                f"block={sample.block_index} has shape {image.shape}, expected "
                f"({image_rows}, {image_columns})"
            )
        if not np.isin(image, (0, 1)).all():
            raise ValueError("Imagery targets must contain only zero and one")
        image = np.asarray(image, dtype=np.int8)
        images.append(image.reshape(-1))
        sample_keys.append((sample.subject_id, sample.trial_number, sample.block_index))
        seeds.append(sample.seed if isinstance(sample, RandomSample) else _NO_RANDOM_SEED)
        sample_types.append(sample.type)
        pattern_ids.append(sample.pattern_id if isinstance(sample, GeometricSample) else -1)
        fingerprints.append(hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())

    y = np.stack(images).astype(np.int8, copy=False)
    subject_ids = np.asarray([key[0] for key in sample_keys], dtype=np.int64)
    trial_numbers = np.asarray([key[1] for key in sample_keys], dtype=np.int64)
    block_indices = np.asarray([key[2] for key in sample_keys], dtype=np.int64)
    seed_array = np.asarray(seeds, dtype=np.int64)
    pattern_id_array = np.asarray(pattern_ids, dtype=np.int64)
    for array in (y, subject_ids, trial_numbers, block_indices, seed_array, pattern_id_array):
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
        sample_types=tuple(sample_types),
        pattern_ids=pattern_id_array,
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
    train_seeds = _random_seed_set(targets.seeds[train])
    test_seeds = _random_seed_set(targets.seeds[test])
    overlapping_fingerprints = tuple(sorted(train_fingerprints & test_fingerprints))
    overlapping_random_fingerprints = tuple(
        sorted(
            _image_fingerprint_set(targets, train, sample_type="random")
            & _image_fingerprint_set(targets, test, sample_type="random")
        )
    )
    overlapping_geometric_pattern_ids = tuple(
        sorted(
            _pattern_id_set(targets, train, sample_type="geometric")
            & _pattern_id_set(targets, test, sample_type="geometric")
        )
    )

    train_positive_counts = targets.y[train].sum(axis=0, dtype=np.int64)
    test_positive_counts = targets.y[test].sum(axis=0, dtype=np.int64)
    train_has_both = (train_positive_counts > 0) & (train_positive_counts < train.size)
    test_has_both = (test_positive_counts > 0) & (test_positive_counts < test.size)
    return LeakageAudit(
        overlapping_subjects=tuple(sorted(set(split.train_subjects) & set(split.test_subjects))),
        overlapping_sample_keys=tuple(sorted(train_keys & test_keys)),
        overlapping_seeds=tuple(sorted(train_seeds & test_seeds)),
        overlapping_image_fingerprints=overlapping_fingerprints,
        train_positive_counts=train_positive_counts,
        test_positive_counts=test_positive_counts,
        all_tasks_have_both_classes=bool(np.all(train_has_both & test_has_both)),
        overlapping_random_image_fingerprints=overlapping_random_fingerprints,
        overlapping_geometric_pattern_ids=overlapping_geometric_pattern_ids,
    )


def create_cross_subject_protocol(
    targets: PixelTargetDataset,
    *,
    config: SubjectSplitConfig,
) -> EvaluationProtocolDefinition:
    split, _ = create_subject_split(targets, config=config)
    eligible_subjects = tuple(
        int(value) for value in np.unique(targets.subject_ids)
    )
    direction = EvaluationDirection(
        protocol="cross-subject",
        name="cross-subject",
        label="cross-subject",
        train_indices=split.train_indices,
        test_indices=split.test_indices,
        train_subjects=split.train_subjects,
        test_subjects=split.test_subjects,
        eligible_subjects=eligible_subjects,
        excluded_subjects=(),
        n_samples=targets.y.shape[0],
    )
    audit = audit_evaluation_direction(targets, direction)
    _validate_protocol_audit(audit, require_both_classes=config.require_both_classes)
    return EvaluationProtocolDefinition(
        protocol="cross-subject",
        label="cross-subject",
        eligible_subjects=eligible_subjects,
        excluded_subjects=(),
        directions=(direction,),
        audits=(audit,),
    )


def create_within_subject_protocol(
    targets: PixelTargetDataset,
    *,
    require_both_classes: bool = True,
) -> EvaluationProtocolDefinition:
    unexpected_trials = tuple(
        int(value)
        for value in np.setdiff1d(
            np.unique(targets.trial_numbers),
            np.asarray([1, 2], dtype=np.int64),
        )
    )
    if unexpected_trials:
        raise ValueError(
            "Within-subject protocol supports only Trial 1 and Trial 2; "
            f"found {unexpected_trials}"
        )
    all_subjects = tuple(int(value) for value in np.unique(targets.subject_ids))
    eligible_subjects = tuple(
        subject
        for subject in all_subjects
        if {1, 2}
        <= set(
            int(value)
            for value in np.unique(
                targets.trial_numbers[targets.subject_ids == subject]
            )
        )
    )
    excluded_subjects = tuple(
        subject for subject in all_subjects if subject not in set(eligible_subjects)
    )
    if not eligible_subjects:
        raise ValueError("Within-subject protocol requires identities with both trials")

    eligible_mask = np.isin(targets.subject_ids, eligible_subjects)
    directions: list[EvaluationDirection] = []
    audits: list[ProtocolLeakageAudit] = []
    for name, train_trial, test_trial in (
        ("trial-1-to-trial-2", 1, 2),
        ("trial-2-to-trial-1", 2, 1),
    ):
        train_indices = np.flatnonzero(
            eligible_mask & (targets.trial_numbers == train_trial)
        ).astype(np.int64, copy=False)
        test_indices = np.flatnonzero(
            eligible_mask & (targets.trial_numbers == test_trial)
        ).astype(np.int64, copy=False)
        train_indices.setflags(write=False)
        test_indices.setflags(write=False)
        direction = EvaluationDirection(
            protocol="within-subject",
            name=name,
            label=f"Trial {train_trial} -> Trial {test_trial}",
            train_indices=train_indices,
            test_indices=test_indices,
            train_subjects=eligible_subjects,
            test_subjects=eligible_subjects,
            eligible_subjects=eligible_subjects,
            excluded_subjects=excluded_subjects,
            n_samples=targets.y.shape[0],
            train_trial=train_trial,
            test_trial=test_trial,
        )
        audit = audit_evaluation_direction(targets, direction)
        _validate_protocol_audit(audit, require_both_classes=require_both_classes)
        directions.append(direction)
        audits.append(audit)

    return EvaluationProtocolDefinition(
        protocol="within-subject",
        label="identity-overlapping bidirectional cross-trial",
        eligible_subjects=eligible_subjects,
        excluded_subjects=excluded_subjects,
        directions=tuple(directions),
        audits=tuple(audits),
    )


def build_evaluation_protocol(
    targets: PixelTargetDataset,
    *,
    protocol: EvaluationProtocol,
    split_config: SubjectSplitConfig,
) -> EvaluationProtocolDefinition:
    if protocol == "cross-subject":
        return create_cross_subject_protocol(targets, config=split_config)
    if protocol == "within-subject":
        return create_within_subject_protocol(
            targets,
            require_both_classes=split_config.require_both_classes,
        )
    raise ValueError(f"Unsupported evaluation protocol: {protocol!r}")


def audit_evaluation_direction(
    targets: PixelTargetDataset,
    direction: EvaluationDirection,
) -> ProtocolLeakageAudit:
    train = direction.train_indices
    test = direction.test_indices
    train_keys = {targets.sample_keys[int(index)] for index in train}
    test_keys = {targets.sample_keys[int(index)] for index in test}
    train_subjects = tuple(int(value) for value in np.unique(targets.subject_ids[train]))
    test_subjects = tuple(int(value) for value in np.unique(targets.subject_ids[test]))
    train_trials = tuple(int(value) for value in np.unique(targets.trial_numbers[train]))
    test_trials = tuple(int(value) for value in np.unique(targets.trial_numbers[test]))
    overlapping_subjects = tuple(sorted(set(train_subjects) & set(test_subjects)))
    overlapping_trials = tuple(sorted(set(train_trials) & set(test_trials)))

    if direction.protocol == "cross-subject":
        subject_contract_satisfied = (
            not overlapping_subjects
            and train_subjects == direction.train_subjects
            and test_subjects == direction.test_subjects
        )
        trial_contract_satisfied = True
    else:
        subject_contract_satisfied = (
            train_subjects == direction.eligible_subjects
            and test_subjects == direction.eligible_subjects
            and overlapping_subjects == direction.eligible_subjects
        )
        trial_contract_satisfied = (
            train_trials == (direction.train_trial,)
            and test_trials == (direction.test_trial,)
            and not overlapping_trials
        )

    train_positive_counts = targets.y[train].sum(axis=0, dtype=np.int64)
    test_positive_counts = targets.y[test].sum(axis=0, dtype=np.int64)
    train_has_both = (train_positive_counts > 0) & (
        train_positive_counts < train.size
    )
    test_has_both = (test_positive_counts > 0) & (
        test_positive_counts < test.size
    )
    train_positive_counts.setflags(write=False)
    test_positive_counts.setflags(write=False)
    return ProtocolLeakageAudit(
        protocol=direction.protocol,
        direction_name=direction.name,
        overlapping_subjects=overlapping_subjects,
        overlapping_sample_keys=tuple(sorted(train_keys & test_keys)),
        overlapping_seeds=tuple(
            sorted(
                _random_seed_set(targets.seeds[train])
                & _random_seed_set(targets.seeds[test])
            )
        ),
        overlapping_image_fingerprints=tuple(
            sorted(
                _image_fingerprint_set(targets, train)
                & _image_fingerprint_set(targets, test)
            )
        ),
        overlapping_trial_numbers=overlapping_trials,
        train_positive_counts=train_positive_counts,
        test_positive_counts=test_positive_counts,
        all_tasks_have_both_classes=bool(np.all(train_has_both & test_has_both)),
        subject_contract_satisfied=subject_contract_satisfied,
        trial_contract_satisfied=trial_contract_satisfied,
        overlapping_random_image_fingerprints=tuple(
            sorted(
                _image_fingerprint_set(targets, train, sample_type="random")
                & _image_fingerprint_set(targets, test, sample_type="random")
            )
        ),
        overlapping_geometric_pattern_ids=tuple(
            sorted(
                _pattern_id_set(targets, train, sample_type="geometric")
                & _pattern_id_set(targets, test, sample_type="geometric")
            )
        ),
    )


def _random_seed_set(values: np.ndarray) -> set[int]:
    return {int(value) for value in values if int(value) != _NO_RANDOM_SEED}


def _image_fingerprint_set(
    targets: PixelTargetDataset,
    indices: np.ndarray,
    *,
    sample_type: TargetSampleType | None = None,
) -> set[str]:
    if sample_type is None:
        return {targets.image_fingerprints[int(index)] for index in indices}
    return {
        targets.image_fingerprints[int(index)]
        for index in indices
        if targets.sample_types[int(index)] == sample_type
    }


def _pattern_id_set(
    targets: PixelTargetDataset,
    indices: np.ndarray,
    *,
    sample_type: TargetSampleType,
) -> set[int]:
    return {
        int(targets.pattern_ids[int(index)])
        for index in indices
        if targets.sample_types[int(index)] == sample_type and int(targets.pattern_ids[int(index)]) >= 0
    }


def _validate_protocol_audit(
    audit: ProtocolLeakageAudit,
    *,
    require_both_classes: bool,
) -> None:
    if audit.has_forbidden_leakage:
        raise ValueError(
            f"Evaluation direction {audit.direction_name!r} violates its leakage contract"
        )
    if require_both_classes and not audit.all_tasks_have_both_classes:
        raise ValueError(
            f"Evaluation direction {audit.direction_name!r} lacks both classes in a pixel task"
        )
