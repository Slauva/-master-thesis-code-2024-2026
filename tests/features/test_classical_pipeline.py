import numpy as np

from features import extract_classical_feature_blocks, load_feature_config


def test_classical_pipeline_crops_resamples_windows_and_selects_groups() -> None:
    sfreq = 1_000.0
    times = np.arange(16_001, dtype=np.float64) / sfreq
    eeg = np.stack(
        (
            np.sin(2.0 * np.pi * 10.0 * times),
            np.sin(2.0 * np.pi * 20.0 * times),
        )
    ).astype(np.float32)
    config = load_feature_config(
        overrides={
            "window_seconds": 4.0,
            "window_stride_seconds": 2.0,
            "feature_groups": ["time", "spectral", "spatial"],
        }
    )

    blocks, bounds = extract_classical_feature_blocks(eeg, source_sfreq=sfreq, config=config)

    assert tuple(block.name for block in blocks) == (
        "time",
        "spectral",
        "covariance",
        "correlation",
        "log_covariance",
    )
    assert blocks[0].values.shape == (6, 2, 13)
    assert blocks[1].values.shape == (6, 2, 14)
    assert all(block.values.shape == (6, 2, 2) for block in blocks[2:])
    assert all(block.values.dtype == np.dtype(np.float32) for block in blocks)
    np.testing.assert_array_equal(bounds[[0, -1]], np.array([[0.5, 4.5], [10.5, 14.5]]))


def test_classical_pipeline_rejects_local_pattern_only_configuration() -> None:
    config = load_feature_config(overrides={"feature_groups": ["local_patterns"]})

    try:
        extract_classical_feature_blocks(
            np.zeros((2, 16_001), dtype=np.float32),
            source_sfreq=1_000.0,
            config=config,
        )
    except ValueError as error:
        assert "No classical feature groups" in str(error)
    else:
        raise AssertionError("Expected local-pattern-only classical extraction to fail")
