import numpy as np
import pytest

from experiments.bnci2014_001 import (
    BNCIEpochMetadata,
    compute_bnci_spectral_sample,
    extract_bnci_feature_set,
    flatten_bnci_feature_set,
    load_bnci_feature_config,
    prepare_bnci_epoch,
    spectral_payload_nbytes,
)


def _metadata() -> BNCIEpochMetadata:
    return BNCIEpochMetadata(
        subject=1,
        session="0train",
        run="0",
        epoch_index=0,
        label="left_hand",
        y=0,
    )


def _epoch() -> np.ndarray:
    sfreq = 250.0
    time = np.arange(1001, dtype=np.float64) / sfreq
    return np.stack(
        (
            np.sin(2.0 * np.pi * 10.0 * time),
            0.5 * np.sin(2.0 * np.pi * 20.0 * time),
        )
    ).astype(np.float32)


def test_prepare_bnci_epoch_trims_moabb_inclusive_endpoint() -> None:
    prepared = prepare_bnci_epoch(
        _epoch(),
        _metadata(),
        eeg_channels=("C3", "C4"),
    )

    assert prepared.eeg.shape == (2, 1000)
    assert prepared.eeg_channels == ("C3", "C4")
    assert prepared.bounds_seconds == (0.0, 4.0)
    np.testing.assert_array_equal(prepared.eeg, _epoch()[:, :1000])


def test_bnci_feature_adapter_extracts_and_flattens_feature_rows() -> None:
    config = load_bnci_feature_config(
        overrides={
            "feature_groups": ["time", "spectral"],
            "window_seconds": 2.0,
            "window_stride_seconds": 2.0,
        }
    )

    feature_set = extract_bnci_feature_set(
        _epoch(),
        _metadata(),
        config=config,
        eeg_channels=("C3", "C4"),
    )
    matrix = flatten_bnci_feature_set(feature_set)

    assert tuple(block.name for block in feature_set.blocks) == ("time", "spectral")
    assert feature_set.window_bounds_seconds.tolist() == [[0.0, 2.0], [2.0, 4.0]]
    assert matrix.X.shape[0] == 2
    assert matrix.sample_keys == (_metadata().sample_key, _metadata().sample_key)
    assert matrix.window_indices.tolist() == [0, 1]
    assert len(matrix.feature_names) == matrix.X.shape[1]
    assert np.isfinite(matrix.X).all()


@pytest.mark.parametrize("method", ["fft", "stft"])
def test_bnci_spectral_adapter_returns_method_specific_shapes(method: str) -> None:
    sample = compute_bnci_spectral_sample(
        _epoch(),
        _metadata(),
        method=method,  # type: ignore[arg-type]
        eeg_channels=("C3", "C4"),
    )

    assert sample.eeg_channels == ("C3", "C4")
    assert sample.frequencies.shape == (39,)
    assert sample.eeg_power.shape[:2] == (2, 39)
    assert sample.eeg_power.dtype == np.float32
    assert sample.epoch_bounds_seconds == (0.0, 4.0)
    assert spectral_payload_nbytes(sample) > sample.eeg_power.nbytes
    if method == "fft":
        assert sample.eeg_power.ndim == 2
        assert sample.times is None
    else:
        assert sample.eeg_power.ndim == 3
        assert sample.times is not None
        assert sample.eeg_power.shape[-1] == sample.times.shape[0]
