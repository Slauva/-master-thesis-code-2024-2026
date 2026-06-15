from pathlib import Path

import numpy as np
import pytest

from experiments.logistic_regression import (
    SubjectSplit,
    audit_subject_split,
    build_random_imagery_targets,
    create_subject_split,
    load_logistic_regression_config,
)
from utils.datasets import GeometricSample, NumpyDataset, RandomSample


def _random_sample(
    *,
    subject: int,
    block: int,
    seed: int,
    value: int,
) -> RandomSample:
    base_image = np.random.default_rng(subject).integers(0, 2, size=(6, 6), dtype=np.int8)
    image = base_image if value % 2 == 0 else 1 - base_image
    return RandomSample(
        subject_id=subject,
        trial_number=1,
        Exec_Block_Index=block,
        eeg_path=Path(f"S_{subject}/random_EEG_{block}.fif"),
        eog_path=Path(f"S_{subject}/random_EOG_{block}.fif"),
        img=image.tolist(),
        seed=seed,
    )


def test_builds_row_major_binary_targets_and_metadata() -> None:
    second = _random_sample(subject=2, block=1, seed=20, value=1)
    first = _random_sample(subject=1, block=2, seed=10, value=0)

    targets = build_random_imagery_targets([second, first])

    assert targets.y.shape == (2, 36)
    assert targets.y.dtype == np.int8
    assert targets.pixel_names[0] == "pixel_r0_c0"
    assert targets.pixel_names[-1] == "pixel_r5_c5"
    assert targets.sample_keys == ((1, 1, 2), (2, 1, 1))
    np.testing.assert_array_equal(targets.y[0], np.asarray(first.img, dtype=np.int8).reshape(-1))
    assert not targets.y.flags.writeable


def test_rejects_non_random_or_invalid_images() -> None:
    geometric = GeometricSample(
        subject_id=1,
        trial_number=1,
        Exec_Block_Index=1,
        eeg_path=Path("eeg.fif"),
        eog_path=Path("eog.fif"),
        img=[[0] * 6 for _ in range(6)],
        pattern_id=1,
    )
    with pytest.raises(TypeError, match="RandomSample"):
        build_random_imagery_targets([geometric])

    invalid = _random_sample(subject=1, block=1, seed=1, value=0).model_copy(
        update={"img": [[0, 1], [1, 0]]}
    )
    with pytest.raises(ValueError, match="shape"):
        build_random_imagery_targets([invalid])


def test_subject_split_is_deterministic_and_disjoint() -> None:
    samples = [
        _random_sample(subject=subject, block=block, seed=subject * 100 + block, value=block)
        for subject in range(1, 11)
        for block in (1, 2)
    ]
    targets = build_random_imagery_targets(samples)
    config = load_logistic_regression_config().split

    first, first_audit = create_subject_split(targets, config=config)
    second, second_audit = create_subject_split(targets, config=config)

    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    assert not set(first.train_subjects) & set(first.test_subjects)
    assert not first_audit.has_leakage
    assert first_audit.all_tasks_have_both_classes
    assert second_audit.all_tasks_have_both_classes


def test_split_audit_detects_seed_and_image_overlap() -> None:
    first = _random_sample(subject=1, block=1, seed=7, value=0)
    second = first.model_copy(
        update={
            "subject_id": 2,
            "eeg_path": Path("S_2/random_EEG_1.fif"),
            "eog_path": Path("S_2/random_EOG_1.fif"),
        }
    )
    samples = [first, second]
    targets = build_random_imagery_targets(samples)
    split = SubjectSplit(
        train_indices=np.asarray([0], dtype=np.int64),
        test_indices=np.asarray([1], dtype=np.int64),
        train_subjects=(1,),
        test_subjects=(2,),
        n_samples=2,
        random_state=42,
        test_size=0.5,
    )

    audit = audit_subject_split(targets, split)

    assert audit.overlapping_seeds == (7,)
    assert len(audit.overlapping_image_fingerprints) == 1
    assert audit.has_leakage
    assert not audit.all_tasks_have_both_classes


def test_real_random_imagery_split_matches_fixed_protocol() -> None:
    config = load_logistic_regression_config()
    dataset = NumpyDataset(
        config.dataset.dataset_dir,
        dataset_step_type=config.dataset.recording_family,
        dataset_pattern_type=config.dataset.pattern_type,
        cache_policy="none",
    )
    targets = build_random_imagery_targets(
        dataset.samples,
        image_rows=config.dataset.image_rows,
        image_columns=config.dataset.image_columns,
    )

    split, audit = create_subject_split(targets, config=config.split)

    assert targets.y.shape == (180, 36)
    assert split.train_indices.size == 141
    assert split.test_indices.size == 39
    assert len(split.train_subjects) == 26
    assert len(split.test_subjects) == 7
    assert split.test_subjects == (9, 10, 16, 18, 20, 28, 33)
    assert not audit.has_leakage
    assert audit.all_tasks_have_both_classes
    assert audit.train_positive_counts.min() == 59
    assert audit.train_positive_counts.max() == 81
    assert audit.test_positive_counts.min() == 14
    assert audit.test_positive_counts.max() == 26
