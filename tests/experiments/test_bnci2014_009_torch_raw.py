from typing import Any

import numpy as np
import pandas as pd
import pytest

from experiments.bnci2014_009 import (
    build_epoch_dataset,
    create_leave_one_subject_splits,
    load_bnci009_config,
    materialize_raw_tensor_dataset,
    run_raw_torch_benchmark,
    run_raw_torch_variant,
    validate_raw_torch_manifest,
    write_raw_torch_benchmark,
)
from experiments.bnci2014_009.torch_raw import (
    apply_tensor_standardizer,
    fit_class_weights,
    fit_tensor_standardizer,
    select_validation_indices,
)


def _toy_dataset() -> Any:
    rng = np.random.default_rng(42)
    rows = []
    labels = []
    epochs = []
    for subject in (1, 2, 3):
        for index in range(12):
            label = "Target" if index % 4 == 0 else "NonTarget"
            signal = rng.normal(size=(8, 64)).astype(np.float32)
            if label == "Target":
                signal[:, 20:30] += 0.5
            rows.append({"subject": subject, "session": "0", "run": "0"})
            labels.append(label)
            epochs.append(signal)
    return build_epoch_dataset(np.asarray(epochs), labels, pd.DataFrame(rows), dtype="float32")


def test_raw_tensor_dataset_contract() -> None:
    dataset = _toy_dataset()
    raw = materialize_raw_tensor_dataset(dataset)

    assert raw.X.shape == (36, 1, 8, 64)
    assert raw.y.tolist() == dataset.y.tolist()
    assert raw.input_shape.tensor_shape == (1, 8, 64)
    assert raw.sample_keys == dataset.sample_keys


def test_raw_torch_validation_split_stays_inside_training_subjects() -> None:
    dataset = _toy_dataset()
    split = create_leave_one_subject_splits(dataset)[0]
    config = load_bnci009_config(
        overrides={
            "dataset": {"subjects": [1, 2, 3]},
            "raw_torch": {"architectures": ["raw-cnn"], "device": "cpu"},
        }
    )

    train_fit_indices, validation_indices, validation_subject = select_validation_indices(
        dataset,
        split,
        split_config=config.split,
    )

    assert validation_subject in split.train_subjects
    assert validation_subject not in split.test_subjects
    assert not set(train_fit_indices.tolist()) & set(validation_indices.tolist())
    assert not set(validation_indices.tolist()) & set(split.test_indices.tolist())


def test_raw_torch_class_weights_are_train_only_balanced() -> None:
    weights = fit_class_weights(
        np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64),
        n_classes=2,
        weighting="balanced",
    )

    assert weights is not None
    assert weights.tolist() == pytest.approx([1.5, 0.75])


def test_raw_torch_standardizer_uses_train_fit_tensor_statistics() -> None:
    train = np.asarray([[[[1.0, 3.0]]], [[[3.0, 7.0]]]], dtype=np.float32)
    test = np.asarray([[[[5.0, 11.0]]]], dtype=np.float32)

    mean, std = fit_tensor_standardizer(train)
    standardized = apply_tensor_standardizer(test, mean=mean, std=std)

    assert mean.tolist() == [[[[2.0, 5.0]]]]
    assert std.tolist() == [[[[1.0, 2.0]]]]
    assert standardized.tolist() == [[[[3.0, 3.0]]]]


def test_run_raw_torch_variant_produces_binary_probabilities() -> None:
    dataset = _toy_dataset()
    raw = materialize_raw_tensor_dataset(dataset)
    config = load_bnci009_config(
        overrides={
            "dataset": {"subjects": [1, 2, 3], "source_sfreq": 64.0},
            "raw_torch": {
                "architectures": ["raw-cnn"],
                "max_epochs": 1,
                "patience": 1,
                "batch_size": 8,
                "device": "cpu",
                "hidden_channels": 4,
            },
        }
    )

    result = run_raw_torch_variant(
        dataset,
        raw,
        architecture="raw-cnn",
        config=config.raw_torch,
        split_config=config.split,
    )

    assert result.model_id == "raw-cnn-raw-erp"
    assert len(result.folds) == 3
    assert result.summary["n_samples"] == 36
    assert all(fold.probabilities.shape == (12, 2) for fold in result.folds)
    assert all(fold.metrics.roc_auc is not None for fold in result.folds)


def test_run_raw_torch_benchmark_subset_and_manifest(tmp_path) -> None:
    dataset = _toy_dataset()
    config = load_bnci009_config(
        overrides={
            "dataset": {"subjects": [1, 2, 3], "source_sfreq": 64.0},
            "raw_torch": {
                "architectures": ["raw-cnn"],
                "max_epochs": 1,
                "patience": 1,
                "batch_size": 8,
                "device": "cpu",
                "hidden_channels": 4,
            },
            "artifacts": {"root": str(tmp_path)},
        }
    )

    result = run_raw_torch_benchmark(config, dataset=dataset)
    run_dir = write_raw_torch_benchmark(config, result)
    validate_raw_torch_manifest(run_dir)

    assert (run_dir / "evaluation.json").is_file()
    assert (run_dir / "training.json").is_file()
    assert (run_dir / "arrays" / "raw_cnn_raw_erp_probabilities.npy").is_file()
