import numpy as np
import pytest

from features import PreparedFeatureWindows, extract_time_features


def _windows(values: np.ndarray) -> PreparedFeatureWindows:
    return PreparedFeatureWindows(
        values=np.asarray(values, dtype=np.float64),
        bounds_seconds=np.array([[0.5, 4.5]], dtype=np.float64),
        sfreq=125.0,
    )


def test_constant_signal_time_features_are_finite_and_explicit() -> None:
    block = extract_time_features(_windows([[[2.0, 2.0, 2.0, 2.0, 2.0]]]))
    values = dict(zip(block.feature_names, block.values[0, 0], strict=True))

    assert values["mean"] == 2.0
    assert values["rms"] == 2.0
    for name in (
        "variance",
        "std",
        "mad",
        "peak_to_peak",
        "skewness",
        "excess_kurtosis",
        "normalized_line_length",
        "zero_crossing_rate",
        "hjorth_mobility",
        "hjorth_complexity",
    ):
        assert values[name] == 0.0


def test_ramp_time_features_match_population_definitions() -> None:
    block = extract_time_features(_windows([[[-2.0, -1.0, 0.0, 1.0, 2.0]]]), dtype=np.float64)
    values = dict(zip(block.feature_names, block.values[0, 0], strict=True))

    assert values["mean"] == pytest.approx(0.0)
    assert values["variance"] == pytest.approx(2.0)
    assert values["std"] == pytest.approx(np.sqrt(2.0))
    assert values["rms"] == pytest.approx(np.sqrt(2.0))
    assert values["median"] == pytest.approx(0.0)
    assert values["mad"] == pytest.approx(1.0)
    assert values["peak_to_peak"] == pytest.approx(4.0)
    assert values["skewness"] == pytest.approx(0.0)
    assert values["excess_kurtosis"] == pytest.approx(-1.3)
    assert values["normalized_line_length"] == pytest.approx(1.0)
    assert values["zero_crossing_rate"] == pytest.approx(0.25)
    assert values["hjorth_mobility"] == pytest.approx(0.0)
    assert values["hjorth_complexity"] == pytest.approx(0.0)


def test_time_features_reject_too_short_windows() -> None:
    with pytest.raises(ValueError, match="at least three"):
        extract_time_features(_windows([[[0.0, 1.0]]]))
