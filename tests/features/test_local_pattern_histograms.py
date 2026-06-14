import numpy as np
import pytest

from features import build_code_histograms


def test_count_histogram_preserves_code_occurrences() -> None:
    codes = np.array([[[0, 1, 1, 255]]], dtype=np.uint32)

    histogram = build_code_histograms(codes, neighbors=8, mode="count", dtype=np.float64)

    assert histogram.shape == (1, 1, 256)
    assert histogram.sum() == 4
    assert histogram[0, 0, 0] == 1
    assert histogram[0, 0, 1] == 2
    assert histogram[0, 0, 255] == 1


def test_probability_histogram_has_unit_mass_per_channel() -> None:
    codes = np.array(
        [
            [[0, 1, 1, 255], [3, 3, 3, 3]],
            [[2, 2, 1, 1], [4, 5, 6, 7]],
        ],
        dtype=np.uint32,
    )

    histogram = build_code_histograms(codes, neighbors=8, mode="probability")

    np.testing.assert_allclose(histogram.sum(axis=-1), np.ones((2, 2)))
    assert histogram.dtype == np.dtype(np.float32)


def test_histogram_rejects_out_of_range_or_non_integer_codes() -> None:
    with pytest.raises(ValueError, match="range"):
        build_code_histograms(np.array([256]), neighbors=8)
    with pytest.raises(TypeError, match="integer"):
        build_code_histograms(np.array([1.5]), neighbors=8)
