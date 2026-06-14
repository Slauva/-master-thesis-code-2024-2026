import numpy as np
import pytest

from features import compute_lbp_codes, compute_lgp_codes, compute_lndp_codes


def test_lndp_reproduces_paper_figure_code_7() -> None:
    paper_segment = np.array([50, 35, 32, 18, 10, 3, 4, 8, 12], dtype=np.float64)

    codes = compute_lndp_codes(paper_segment, neighbors=8)

    np.testing.assert_array_equal(codes, np.array([7], dtype=np.uint32))


def test_lgp_reproduces_paper_figure_code_224() -> None:
    paper_segment = np.array([50, 35, 32, 18, 10, 3, -1, -5, -6], dtype=np.float64)

    codes = compute_lgp_codes(paper_segment, neighbors=8)

    np.testing.assert_array_equal(codes, np.array([224], dtype=np.uint32))


def test_code_count_uses_full_neighborhoods_without_padding() -> None:
    signal = np.arange(20, dtype=np.float64)

    for compute_codes in (compute_lndp_codes, compute_lgp_codes, compute_lbp_codes):
        codes = compute_codes(signal, neighbors=8)
        assert codes.shape == (12,)


def test_code_functions_preserve_leading_axes() -> None:
    signal = np.arange(2 * 3 * 20, dtype=np.float64).reshape(2, 3, 20)

    for compute_codes in (compute_lndp_codes, compute_lgp_codes, compute_lbp_codes):
        codes = compute_codes(signal, neighbors=8)
        assert codes.shape == (2, 3, 12)
        assert codes.dtype == np.dtype(np.uint32)


def test_global_offset_does_not_change_local_pattern_codes() -> None:
    rng = np.random.default_rng(17)
    signal = rng.normal(size=100)

    for compute_codes in (compute_lndp_codes, compute_lgp_codes, compute_lbp_codes):
        np.testing.assert_array_equal(
            compute_codes(signal, neighbors=8),
            compute_codes(signal + 500.0, neighbors=8),
        )


@pytest.mark.parametrize("neighbors", [1, 3, 18])
def test_code_functions_reject_invalid_neighbor_count(neighbors: int) -> None:
    with pytest.raises(ValueError, match="even integer"):
        compute_lndp_codes(np.arange(20), neighbors=neighbors)


def test_code_functions_reject_short_or_non_finite_signal() -> None:
    with pytest.raises(ValueError, match="at least 9"):
        compute_lgp_codes(np.arange(8), neighbors=8)
    with pytest.raises(ValueError, match="finite"):
        compute_lbp_codes(np.array([0.0] * 8 + [np.nan]), neighbors=8)
