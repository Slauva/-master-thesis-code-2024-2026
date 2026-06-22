import numpy as np
import pandas as pd
import pytest

from experiments.bnci2014_001 import (
    BNCISplit,
    audit_split,
    build_epoch_dataset,
    create_leave_one_subject_splits,
    load_bnci_config,
)


def _toy_epoch_inputs() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    rows = []
    labels = []
    for subject in (1, 2, 3):
        for session in ("0train", "1test"):
            for run in ("0", "1"):
                for label in ("left_hand", "right_hand", "feet", "tongue"):
                    rows.append({"subject": subject, "session": session, "run": run})
                    labels.append(label)
    X = np.arange(len(rows) * 2 * 5, dtype=np.float64).reshape(len(rows), 2, 5)
    return X, labels, pd.DataFrame(rows)


def test_load_bnci_config_validates_default_contract() -> None:
    config = load_bnci_config()

    assert config.dataset.dataset_code == "BNCI2014-001"
    assert config.dataset.subjects == tuple(range(1, 10))
    assert config.dataset.labels == ("left_hand", "right_hand", "feet", "tongue")
    assert config.dataset.epoch_start_seconds == 2.0
    assert config.dataset.epoch_end_seconds == 6.0
    assert config.split.primary_protocol == "leave-one-subject-out"
    assert config.split.group_by == "subject"


def test_build_epoch_dataset_preserves_deterministic_metadata_and_targets() -> None:
    X, labels, metadata = _toy_epoch_inputs()

    dataset = build_epoch_dataset(X, labels, metadata)

    assert dataset.X.shape == (48, 2, 5)
    assert dataset.X.dtype == np.float32
    assert dataset.y.dtype == np.int64
    assert dataset.class_names == ("left_hand", "right_hand", "feet", "tongue")
    assert dataset.metadata[0].sample_key == (1, "0train", "0", 0)
    assert dataset.metadata[3].sample_key == (1, "0train", "0", 3)
    assert dataset.metadata[4].sample_key == (1, "0train", "1", 0)
    np.testing.assert_array_equal(dataset.y[:4], np.asarray([0, 1, 2, 3], dtype=np.int64))
    assert not dataset.X.flags.writeable
    assert not dataset.y.flags.writeable


def test_build_epoch_dataset_rejects_missing_metadata_and_unknown_labels() -> None:
    X, labels, metadata = _toy_epoch_inputs()

    with pytest.raises(ValueError, match="missing required"):
        build_epoch_dataset(X, labels, metadata.drop(columns=["run"]))

    bad_labels = list(labels)
    bad_labels[0] = "rest"
    with pytest.raises(ValueError, match="Unknown BNCI labels"):
        build_epoch_dataset(X, bad_labels, metadata)


def test_leave_one_subject_splits_are_disjoint_and_class_complete() -> None:
    X, labels, metadata = _toy_epoch_inputs()
    dataset = build_epoch_dataset(X, labels, metadata)

    splits = create_leave_one_subject_splits(dataset)

    assert len(splits) == 3
    assert tuple(split.test_subjects for split in splits) == ((1,), (2,), (3,))
    for split in splits:
        audit = audit_split(dataset, split)
        assert not audit.has_forbidden_leakage
        assert not audit.overlapping_subjects
        assert not audit.overlapping_sample_keys
        assert audit.all_train_classes_present
        assert audit.all_test_classes_present
        assert audit.test_class_counts == {
            "left_hand": 4,
            "right_hand": 4,
            "feet": 4,
            "tongue": 4,
        }


def test_split_audit_detects_subject_and_sample_key_leakage() -> None:
    X, labels, metadata = _toy_epoch_inputs()
    dataset = build_epoch_dataset(X, labels, metadata)
    leaky = BNCISplit(
        name="leaky",
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        test_indices=np.asarray([4, 5, 6, 7], dtype=np.int64),
        train_subjects=(1,),
        test_subjects=(1,),
        n_samples=len(dataset.metadata),
    )

    audit = audit_split(dataset, leaky)

    assert audit.overlapping_subjects == (1,)
    assert audit.has_subject_leakage
    assert audit.has_forbidden_leakage

    with pytest.raises(ValueError, match="disjoint"):
        BNCISplit(
            name="overlap",
            train_indices=np.asarray([0, 1], dtype=np.int64),
            test_indices=np.asarray([1, 2], dtype=np.int64),
            train_subjects=(1,),
            test_subjects=(2,),
            n_samples=len(dataset.metadata),
        )
