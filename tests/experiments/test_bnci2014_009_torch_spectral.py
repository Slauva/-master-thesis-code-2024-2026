from typing import Any

import numpy as np
import pandas as pd

from experiments.bnci2014_009 import (
    build_epoch_dataset,
    load_bnci009_config,
    materialize_fft_tensor_dataset,
    run_spectral_torch_benchmark,
    run_spectral_torch_variant,
    validate_spectral_torch_manifest,
    write_spectral_torch_benchmark,
)


def _toy_dataset() -> Any:
    rng = np.random.default_rng(42)
    rows = []
    labels = []
    epochs = []
    t = np.arange(128, dtype=np.float32) / 128.0
    target_wave = np.sin(2 * np.pi * 12.0 * t)
    non_target_wave = np.sin(2 * np.pi * 6.0 * t)
    for subject in (1, 2, 3):
        for index in range(12):
            label = "Target" if index % 4 == 0 else "NonTarget"
            carrier = target_wave if label == "Target" else non_target_wave
            signal = rng.normal(scale=0.05, size=(8, 128)).astype(np.float32)
            signal += carrier[np.newaxis, :].astype(np.float32)
            rows.append({"subject": subject, "session": "0", "run": "0"})
            labels.append(label)
            epochs.append(signal)
    return build_epoch_dataset(np.asarray(epochs), labels, pd.DataFrame(rows), dtype="float32")


def test_fft_tensor_dataset_contract() -> None:
    dataset = _toy_dataset()
    spectral = materialize_fft_tensor_dataset(dataset, source_sfreq=128.0)

    assert spectral.method == "fft"
    assert spectral.X.shape == (36, 1, 8, 39)
    assert spectral.input_shape.tensor_shape == (1, 8, 39)
    assert spectral.frequencies[0] == 2.0
    assert spectral.frequencies[-1] == 40.0
    assert spectral.sample_keys == dataset.sample_keys


def test_spectral_torch_variant_produces_aligned_binary_probabilities() -> None:
    dataset = _toy_dataset()
    spectral = materialize_fft_tensor_dataset(dataset, source_sfreq=128.0)
    config = load_bnci009_config(
        overrides={
            "dataset": {"subjects": [1, 2, 3], "source_sfreq": 128.0},
            "spectral_torch": {
                "architectures": ["eegnet"],
                "spectral_methods": ["fft"],
                "max_epochs": 1,
                "patience": 1,
                "batch_size": 8,
                "device": "cpu",
            },
        }
    )

    result = run_spectral_torch_variant(
        dataset,
        spectral,
        architecture="eegnet",
        config=config.spectral_torch,
        split_config=config.split,
    )

    assert result.model_id == "eegnet-fft-spectral"
    assert len(result.folds) == 3
    assert result.summary["n_samples"] == 36
    assert all(fold.probabilities.shape == (12, 2) for fold in result.folds)


def test_spectral_torch_benchmark_subset_and_manifest(tmp_path) -> None:
    dataset = _toy_dataset()
    config = load_bnci009_config(
        overrides={
            "dataset": {"subjects": [1, 2, 3], "source_sfreq": 128.0},
            "spectral_torch": {
                "architectures": ["eegnet"],
                "spectral_methods": ["fft"],
                "max_epochs": 1,
                "patience": 1,
                "batch_size": 8,
                "device": "cpu",
            },
            "artifacts": {"root": str(tmp_path)},
        }
    )

    result = run_spectral_torch_benchmark(config, dataset=dataset)
    run_dir = write_spectral_torch_benchmark(config, result)
    validate_spectral_torch_manifest(run_dir)

    comparison = (run_dir / "comparison.json").read_text(encoding="utf-8")
    assert "deferred_methods" in comparison
    assert (run_dir / "arrays" / "eegnet_fft_spectral_probabilities.npy").is_file()
