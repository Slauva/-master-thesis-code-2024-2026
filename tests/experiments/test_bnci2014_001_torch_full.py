from typing import Any

import numpy as np
import pandas as pd
import pytest

from experiments.bnci2014_001 import (
    BNCISpectralTensorDataset,
    build_epoch_dataset,
    load_bnci_config,
    materialize_spectral_tensor_dataset,
    run_full_torch_variant,
)
from experiments.random_imagery_torch.models import SpectralModelShape


def _toy_bnci_dataset() -> Any:
    rows = []
    labels = []
    for subject in (1, 2, 3):
        for label in ("left_hand", "right_hand", "feet", "tongue"):
            rows.append({"subject": subject, "session": "0train", "run": "0"})
            labels.append(label)
    X = np.zeros((len(rows), 4, 1001), dtype=np.float32)
    return build_epoch_dataset(X, labels, pd.DataFrame(rows))


def test_full_torch_spectral_tensor_dataset_contract() -> None:
    dataset = _toy_bnci_dataset()
    X = np.ones((12, 1, 4, 5), dtype=np.float32)
    y = np.asarray(dataset.y, dtype=np.int64)
    spectral = BNCISpectralTensorDataset(
        method="fft",
        X=X,
        y=y,
        sample_keys=dataset.sample_keys,
        input_shape=SpectralModelShape(input_planes=1, electrodes=4, width=5),
        tensor_transform="log1p_nonnegative_power",
    )

    assert spectral.input_shape.tensor_shape == (1, 4, 5)
    assert spectral.sample_keys == dataset.sample_keys


def test_full_torch_materialization_reuses_tensor_chunks(tmp_path) -> None:
    dataset = _toy_bnci_dataset()
    first = materialize_spectral_tensor_dataset(
        dataset,
        method="fft",
        source_sfreq=250.0,
        chunk_dir=tmp_path / "chunks",
        chunk_size=5,
    )
    second = materialize_spectral_tensor_dataset(
        dataset,
        method="fft",
        source_sfreq=250.0,
        chunk_dir=tmp_path / "chunks",
        chunk_size=5,
    )

    assert first.X.shape == second.X.shape == (12, 1, 4, 39)
    assert first.input_shape == second.input_shape
    assert np.array_equal(first.X, second.X)
    assert len(sorted((tmp_path / "chunks").glob("*.npy"))) == 3


def test_full_torch_variant_uses_four_class_head() -> None:
    dataset = _toy_bnci_dataset()
    config = load_bnci_config(
        overrides={
            "dataset": {"subjects": [1, 2, 3]},
            "torch_full": {
                "architectures": ["eegnet"],
                "spectral_methods": ["fft"],
                "max_epochs": 1,
                "patience": 1,
                "batch_size": 4,
                "device": "cpu",
            },
        }
    )
    spectral = BNCISpectralTensorDataset(
        method="fft",
        X=np.random.default_rng(42).normal(size=(12, 1, 4, 39)).astype(np.float32),
        y=np.asarray(dataset.y, dtype=np.int64),
        sample_keys=dataset.sample_keys,
        input_shape=SpectralModelShape(input_planes=1, electrodes=4, width=39),
        tensor_transform="log1p_nonnegative_power",
    )

    result = run_full_torch_variant(
        dataset,
        spectral,
        architecture="eegnet",
        config=config.torch_full,
        split_config=config.split,
    )

    assert result.model_id == "eegnet-fft-bnci"
    assert len(result.folds) == 3
    assert result.summary["n_samples"] == 12
    assert all(fold.probabilities.shape == (4, 4) for fold in result.folds)


def test_full_torch_variant_accepts_flattened_time_frequency_input() -> None:
    dataset = _toy_bnci_dataset()
    config = load_bnci_config(
        overrides={
            "dataset": {"subjects": [1, 2, 3]},
            "torch_full": {
                "architectures": ["eegnet"],
                "spectral_methods": ["morlet"],
                "max_epochs": 1,
                "patience": 1,
                "batch_size": 4,
                "device": "cpu",
            },
        }
    )
    spectral = BNCISpectralTensorDataset(
        method="morlet",
        X=np.random.default_rng(42).normal(size=(12, 1, 4, 234)).astype(np.float32),
        y=np.asarray(dataset.y, dtype=np.int64),
        sample_keys=dataset.sample_keys,
        input_shape=SpectralModelShape(input_planes=1, electrodes=4, width=234),
        tensor_transform="log1p_nonnegative_power_flat_tf_v2",
    )

    result = run_full_torch_variant(
        dataset,
        spectral,
        architecture="eegnet",
        config=config.torch_full,
        split_config=config.split,
    )

    assert result.model_id == "eegnet-morlet-bnci"
    assert all(fold.probabilities.shape == (4, 4) for fold in result.folds)


def test_full_torch_rejects_inconsistent_tensor_sample_count() -> None:
    dataset = _toy_bnci_dataset()

    with pytest.raises(ValueError):
        BNCISpectralTensorDataset(
            method="fft",
            X=np.ones((11, 1, 4, 5), dtype=np.float32),
            y=np.asarray(dataset.y, dtype=np.int64),
            sample_keys=dataset.sample_keys,
            input_shape=SpectralModelShape(input_planes=1, electrodes=4, width=5),
            tensor_transform="log1p_nonnegative_power",
        )
