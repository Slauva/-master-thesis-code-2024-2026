from pathlib import Path

import numpy as np
import pytest

from experiments.logistic_regression import (
    SubjectSplit,
    audit_evaluation_direction,
    audit_subject_split,
    build_evaluation_protocol,
    build_random_imagery_targets,
    create_subject_split,
    create_within_subject_protocol,
    load_logistic_regression_config,
)
from experiments.random_imagery.full_dataset_audit import build_full_dataset_audit
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


def _geometric_sample(
    *,
    subject: int,
    trial: int = 1,
    block: int,
    pattern_id: int,
    value: int,
) -> GeometricSample:
    base_image = np.random.default_rng(subject + pattern_id).integers(0, 2, size=(6, 6), dtype=np.int8)
    image = base_image if value % 2 == 0 else 1 - base_image
    return GeometricSample(
        subject_id=subject,
        trial_number=trial,
        Exec_Block_Index=block,
        eeg_path=Path(f"S_{subject}/geometric_EEG_{block}.fif"),
        eog_path=Path(f"S_{subject}/geometric_EOG_{block}.fif"),
        img=image.tolist(),
        pattern_id=pattern_id,
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


def test_builds_mixed_geometric_and_random_targets_when_explicitly_allowed() -> None:
    geometric = _geometric_sample(subject=1, block=1, pattern_id=7, value=0)
    random = _random_sample(subject=2, block=2, seed=42, value=1)

    targets = build_random_imagery_targets(
        [random, geometric],
        allowed_sample_types=("geometric", "random"),
    )

    assert targets.y.shape == (2, 36)
    assert targets.sample_keys == ((1, 1, 1), (2, 1, 2))
    assert targets.seeds.tolist() == [-1, 42]
    assert targets.sample_types == ("geometric", "random")
    assert targets.pattern_ids.tolist() == [7, -1]
    np.testing.assert_array_equal(
        targets.y[0],
        np.asarray(geometric.img, dtype=np.int8).reshape(-1),
    )
    np.testing.assert_array_equal(
        targets.y[1],
        np.asarray(random.img, dtype=np.int8).reshape(-1),
    )


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
    assert len(audit.overlapping_random_image_fingerprints) == 1
    assert audit.has_leakage
    assert not audit.all_tasks_have_both_classes


def test_split_audit_allows_repeated_geometric_patterns_as_labels() -> None:
    image = (np.arange(36, dtype=np.int8).reshape(6, 6) % 2).tolist()
    first = _geometric_sample(subject=1, block=1, pattern_id=3, value=0).model_copy(
        update={"img": image}
    )
    second = _geometric_sample(subject=2, block=1, pattern_id=3, value=0).model_copy(
        update={"img": image}
    )
    targets = build_random_imagery_targets(
        [first, second],
        allowed_sample_types=("geometric",),
    )
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

    assert len(audit.overlapping_image_fingerprints) == 1
    assert audit.overlapping_geometric_pattern_ids == (3,)
    assert not audit.overlapping_random_image_fingerprints
    assert not audit.has_leakage


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


def test_full_imagery_targets_and_protocols_include_geometric_and_random_rows() -> None:
    config = load_logistic_regression_config(
        overrides={
            "dataset": {"pattern_type": None},
        }
    )
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
        allowed_sample_types=config.dataset.target_sample_types,
    )

    assert targets.y.shape == (540, 36)
    assert np.count_nonzero(targets.seeds < 0) == 360
    assert np.count_nonzero(targets.seeds >= 0) == 180

    cross_subject = build_evaluation_protocol(
        targets,
        protocol="cross-subject",
        split_config=config.split,
    )
    within_subject = build_evaluation_protocol(
        targets,
        protocol="within-subject",
        split_config=config.split,
    )

    cross_direction = cross_subject.directions[0]
    assert cross_direction.train_indices.size == 423
    assert cross_direction.test_indices.size == 117
    assert len(cross_direction.train_subjects) == 26
    assert len(cross_direction.test_subjects) == 7
    assert cross_direction.test_subjects == (9, 10, 16, 18, 20, 28, 33)
    assert not cross_subject.audits[0].has_forbidden_leakage
    assert cross_subject.audits[0].all_tasks_have_both_classes
    assert len(cross_subject.audits[0].overlapping_image_fingerprints) == 13
    assert not cross_subject.audits[0].overlapping_random_image_fingerprints
    assert cross_subject.audits[0].overlapping_geometric_pattern_ids == tuple(range(13))

    assert len(within_subject.eligible_subjects) == 27
    assert within_subject.excluded_subjects == (14, 24, 27, 28, 29, 32)
    assert tuple(direction.train_indices.size for direction in within_subject.directions) == (243, 243)
    assert tuple(direction.test_indices.size for direction in within_subject.directions) == (243, 243)
    for audit in within_subject.audits:
        assert audit.all_tasks_have_both_classes
        assert not audit.has_forbidden_leakage
        assert len(audit.overlapping_image_fingerprints) == 13
        assert not audit.overlapping_random_image_fingerprints
        assert audit.overlapping_geometric_pattern_ids == tuple(range(13))


def test_full_dataset_audit_reports_expected_real_corpus_counts() -> None:
    audit = build_full_dataset_audit()

    assert audit["stage1_status"] == "ready"
    assert audit["dataset"]["n_rows"] == 540
    assert audit["dataset"]["sample_type_counts"] == {"geometric": 360, "random": 180}
    assert audit["dataset"]["n_subjects"] == 33
    assert audit["dataset"]["n_subject_trials"] == 60
    assert audit["targets"]["n_rows"] == 540
    assert audit["targets"]["n_pixels"] == 36
    assert audit["targets"]["random_seed_rows"] == 180
    assert audit["targets"]["missing_random_seed_rows"] == 360

    cross_direction = audit["protocols"]["cross_subject"]["directions"][0]
    assert cross_direction["n_train_rows"] == 423
    assert cross_direction["n_test_rows"] == 117
    assert cross_direction["audit"]["all_tasks_have_both_classes"]
    assert not cross_direction["audit"]["has_forbidden_leakage"]
    assert len(cross_direction["audit"]["overlapping_image_fingerprints"]) == 13
    assert not cross_direction["audit"]["overlapping_random_image_fingerprints"]
    assert cross_direction["audit"]["overlapping_geometric_pattern_ids"] == list(range(13))
    assert not cross_direction["audit"]["overlapping_seeds"]

    within_directions = audit["protocols"]["within_subject"]["directions"]
    assert [direction["n_train_rows"] for direction in within_directions] == [243, 243]
    assert [direction["n_test_rows"] for direction in within_directions] == [243, 243]
    assert all(direction["audit"]["all_tasks_have_both_classes"] for direction in within_directions)
    assert not any(direction["audit"]["has_forbidden_leakage"] for direction in within_directions)
    assert [len(direction["audit"]["overlapping_image_fingerprints"]) for direction in within_directions] == [
        13,
        13,
    ]
    assert [
        direction["audit"]["overlapping_geometric_pattern_ids"]
        for direction in within_directions
    ] == [list(range(13)), list(range(13))]
    assert not any(
        direction["audit"]["overlapping_random_image_fingerprints"]
        for direction in within_directions
    )


def test_real_evaluation_protocols_match_fixed_counts_and_leakage_contracts() -> None:
    config = load_logistic_regression_config()
    dataset = NumpyDataset(
        config.dataset.dataset_dir,
        dataset_step_type=config.dataset.recording_family,
        dataset_pattern_type=config.dataset.pattern_type,
        cache_policy="none",
    )
    targets = build_random_imagery_targets(dataset.samples)

    cross_subject = build_evaluation_protocol(
        targets,
        protocol="cross-subject",
        split_config=config.split,
    )
    within_subject = build_evaluation_protocol(
        targets,
        protocol="within-subject",
        split_config=config.split,
    )

    cross_direction = cross_subject.directions[0]
    assert cross_direction.train_indices.size == 141
    assert cross_direction.test_indices.size == 39
    assert len(cross_direction.train_subjects) == 26
    assert len(cross_direction.test_subjects) == 7
    assert not cross_subject.audits[0].has_forbidden_leakage

    assert within_subject.label == "identity-overlapping bidirectional cross-trial"
    assert len(within_subject.eligible_subjects) == 27
    assert within_subject.excluded_subjects == (14, 24, 27, 28, 29, 32)
    assert tuple(direction.train_indices.size for direction in within_subject.directions) == (81, 81)
    assert tuple(direction.test_indices.size for direction in within_subject.directions) == (81, 81)
    for direction, audit in zip(
        within_subject.directions,
        within_subject.audits,
        strict=True,
    ):
        assert direction.train_subjects == within_subject.eligible_subjects
        assert direction.test_subjects == within_subject.eligible_subjects
        assert audit.overlapping_subjects == within_subject.eligible_subjects
        assert not audit.overlapping_sample_keys
        assert not audit.overlapping_seeds
        assert not audit.overlapping_image_fingerprints
        assert not audit.overlapping_trial_numbers
        assert audit.all_tasks_have_both_classes
        assert not audit.has_forbidden_leakage
        repeated_audit = audit_evaluation_direction(targets, direction)
        assert repeated_audit.direction_name == audit.direction_name
        assert repeated_audit.overlapping_subjects == audit.overlapping_subjects
        np.testing.assert_array_equal(
            repeated_audit.train_positive_counts,
            audit.train_positive_counts,
        )
        np.testing.assert_array_equal(
            repeated_audit.test_positive_counts,
            audit.test_positive_counts,
        )


def test_within_subject_protocol_rejects_cross_trial_seed_and_image_leakage() -> None:
    samples = [
        _random_sample(subject=subject, block=trial, seed=subject, value=0).model_copy(
            update={"trial_number": trial}
        )
        for subject in range(1, 7)
        for trial in (1, 2)
    ]
    targets = build_random_imagery_targets(samples)

    with pytest.raises(ValueError, match="leakage contract"):
        create_within_subject_protocol(targets)
