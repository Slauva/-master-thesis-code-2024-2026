import numpy as np

from features import PreparedFeatureWindows, extract_spatial_features


def test_spatial_features_are_symmetric_finite_and_log_consistent() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=1_000)
    signals = np.stack((source, 0.8 * source + 0.2 * rng.normal(size=1_000)))
    windows = PreparedFeatureWindows(
        values=signals[np.newaxis, ...],
        bounds_seconds=np.array([[0.5, 8.5]], dtype=np.float64),
        sfreq=125.0,
    )

    covariance_block, correlation_block, log_covariance_block = extract_spatial_features(
        windows,
        dtype=np.float64,
    )
    covariance = covariance_block.values[0]
    correlation = correlation_block.values[0]
    log_covariance = log_covariance_block.values[0]

    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    np.testing.assert_allclose(correlation, correlation.T, atol=1e-12)
    np.testing.assert_allclose(log_covariance, log_covariance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(correlation), np.ones(2), atol=1e-12)
    assert correlation[0, 1] > 0.9
    assert np.linalg.eigvalsh(covariance).min() > 0

    eigenvalues, eigenvectors = np.linalg.eigh(log_covariance)
    reconstructed = (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T
    np.testing.assert_allclose(reconstructed, covariance, rtol=1e-10, atol=1e-10)


def test_constant_spatial_features_use_finite_zero_sentinel() -> None:
    windows = PreparedFeatureWindows(
        values=np.ones((1, 3, 100), dtype=np.float64),
        bounds_seconds=np.array([[0.5, 1.3]], dtype=np.float64),
        sfreq=125.0,
    )

    blocks = extract_spatial_features(windows)

    for block in blocks:
        np.testing.assert_array_equal(block.values, np.zeros_like(block.values))
