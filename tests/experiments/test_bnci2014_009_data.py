import numpy as np
import pandas as pd
import pytest

from experiments.bnci2014_009 import (
    BNCI009Split,
    audit_split,
    build_epoch_dataset,
    create_leave_one_subject_splits,
    load_bnci009_config,
)


def _toy_epoch_inputs() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    rows = []
    labels = []
    for subject in (1, 2, 3):
        for session in ("0", "1"):
            for run in ("0",):
                for _sequence in range(2):
                    for label in ("Target", "NonTarget", "NonTarget", "NonTarget", "NonTarget", "NonTarget"):
                        rows.append({"subject": subject, "session": session, "run": run})
                        labels.append(label)
    X = np.arange(len(rows) * 4 * 6, dtype=np.float64).reshape(len(rows), 4, 6)
    return X, labels, pd.DataFrame(rows)


def test_load_bnci009_config_validates_default_contract() -> None:
    config = load_bnci009_config()

    assert config.dataset.dataset_code == "BNCI2014-009"
    assert config.dataset.subjects == tuple(range(1, 11))
    assert config.dataset.labels == ("Target", "NonTarget")
    assert config.dataset.epoch_start_seconds == 0.0
    assert config.dataset.epoch_end_seconds == 0.8
    assert config.dataset.source_sfreq == 256.0
    assert config.dataset.n_classes == 2
    assert config.dataset.moabb_filter_low_hz == 1.0
    assert config.dataset.moabb_filter_high_hz == 24.0
    assert config.split.primary_protocol == "leave-one-subject-out"
    assert config.split.group_by == "subject"


def test_build_epoch_dataset_preserves_deterministic_metadata_targets_and_imbalance() -> None:
    X, labels, metadata = _toy_epoch_inputs()

    dataset = build_epoch_dataset(X, labels, metadata)

    assert dataset.X.shape == (72, 4, 6)
    assert dataset.X.dtype == np.float32
    assert dataset.y.dtype == np.int64
    assert dataset.class_names == ("Target", "NonTarget")
    assert dataset.metadata[0].sample_key == (1, "0", "0", 0)
    assert dataset.metadata[5].sample_key == (1, "0", "0", 5)
    assert dataset.metadata[6].sample_key == (1, "0", "0", 6)
    np.testing.assert_array_equal(dataset.y[:6], np.asarray([0, 1, 1, 1, 1, 1], dtype=np.int64))
    assert int(np.sum(dataset.y == 0)) == 12
    assert int(np.sum(dataset.y == 1)) == 60
    assert not dataset.X.flags.writeable
    assert not dataset.y.flags.writeable


def test_build_epoch_dataset_rejects_missing_metadata_and_unknown_labels() -> None:
    X, labels, metadata = _toy_epoch_inputs()

    with pytest.raises(ValueError, match="missing required"):
        build_epoch_dataset(X, labels, metadata.drop(columns=["run"]))

    bad_labels = list(labels)
    bad_labels[0] = "Distractor"
    with pytest.raises(ValueError, match="Unknown BNCI2014_009 labels"):
        build_epoch_dataset(X, bad_labels, metadata)


def test_leave_one_subject_splits_are_disjoint_class_complete_and_imbalance_visible() -> None:
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
        assert audit.test_class_counts == {"Target": 4, "NonTarget": 20}
        assert audit.test_target_fraction == pytest.approx(1 / 6)
        assert audit.train_target_fraction == pytest.approx(1 / 6)


def test_split_audit_detects_subject_and_sample_key_leakage() -> None:
    X, labels, metadata = _toy_epoch_inputs()
    dataset = build_epoch_dataset(X, labels, metadata)
    leaky = BNCI009Split(
        name="leaky",
        train_indices=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
        test_indices=np.asarray([6, 7, 8, 9, 10, 11], dtype=np.int64),
        train_subjects=(1,),
        test_subjects=(1,),
        n_samples=len(dataset.metadata),
    )

    audit = audit_split(dataset, leaky)

    assert audit.overlapping_subjects == (1,)
    assert audit.has_subject_leakage
    assert audit.has_forbidden_leakage

    with pytest.raises(ValueError, match="disjoint"):
        BNCI009Split(
            name="overlap",
            train_indices=np.asarray([0, 1], dtype=np.int64),
            test_indices=np.asarray([1, 2], dtype=np.int64),
            train_subjects=(1,),
            test_subjects=(2,),
            n_samples=len(dataset.metadata),
        )
