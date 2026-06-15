from pathlib import Path

import pytest

from experiments.logistic_regression import (
    run_evaluation_protocol as run_logistic_protocol,
)
from experiments.logistic_regression import write_protocol_evaluation_runs
from experiments.random_imagery import (
    LinearSVMBackend,
    PLSRegressionMultiOutputBackend,
    compare_protocol_models,
    load_calibrated_classifier_config,
    load_regression_config,
    run_model_evaluation_protocol,
    write_model_protocol_runs,
)
from tests.experiments.test_logistic_regression_runner import (
    _protocol_inputs,
    _runner_config,
)


def _write_reference(tmp_path: Path, protocol: str) -> tuple[Path, ...]:
    dataset, targets = _protocol_inputs()
    base = _runner_config()
    config = base.model_copy(
        update={
            "bootstrap_iterations": 20,
            "artifacts": base.artifacts.model_copy(
                update={"root": tmp_path / "reference"}
            ),
        }
    )
    result = run_logistic_protocol(
        protocol,
        config=config,
        dataset=dataset,
        targets=targets,
    )
    return write_protocol_evaluation_runs(
        result,
        targets=targets,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )


def _write_classifier(tmp_path: Path, protocol: str) -> tuple[Path, ...]:
    dataset, targets = _protocol_inputs()
    config = load_calibrated_classifier_config(
        "linear-svm-independent",
        overrides={
            "cross_validation": {"n_splits": 3},
            "feature_screening": {"select_k": 2, "candidates": [["time"]]},
            "grid_search": {
                "select_k": [2],
                "c_values": [1.0],
                "class_weights": ["balanced"],
                "n_jobs": 1,
            },
            "bootstrap_iterations": 20,
            "artifacts": {"root": str(tmp_path / "models")},
        },
    )
    result = run_model_evaluation_protocol(
        protocol,
        config=config,
        backend=LinearSVMBackend(),
        dataset=dataset,
        targets=targets,
    )
    return write_model_protocol_runs(
        result,
        targets=targets,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )


def _write_regressor(tmp_path: Path, protocol: str) -> tuple[Path, ...]:
    dataset, targets = _protocol_inputs()
    config = load_regression_config(
        "pls-regression-multioutput",
        overrides={
            "cross_validation": {"n_splits": 3},
            "feature_screening": {"select_k": 2, "candidates": [["time"]]},
            "grid_search": {
                "select_k": [2],
                "n_components": [1],
                "n_jobs": 1,
            },
            "bootstrap_iterations": 20,
            "artifacts": {"root": str(tmp_path / "models")},
        },
    )
    result = run_model_evaluation_protocol(
        protocol,
        config=config,
        backend=PLSRegressionMultiOutputBackend(),
        dataset=dataset,
        targets=targets,
    )
    return write_model_protocol_runs(
        result,
        targets=targets,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )


@pytest.mark.parametrize(
    ("protocol", "expected_rows"),
    [("cross-subject", 18), ("within-subject", 72)],
)
def test_protocol_comparison_uses_paired_subject_bootstrap(
    tmp_path: Path,
    protocol: str,
    expected_rows: int,
) -> None:
    reference = _write_reference(tmp_path, protocol)
    classifier = _write_classifier(tmp_path, protocol)
    regressor = _write_regressor(tmp_path, protocol)

    comparison = compare_protocol_models(
        protocol,
        reference_run_dirs=reference,
        model_run_dirs={
            "linear-svm-independent": classifier,
            "pls-regression-multioutput": regressor,
        },
        n_resamples=25,
        random_state=7,
        calibration_bins=5,
    )

    assert comparison.n_test_rows == expected_rows
    assert comparison.n_resamples == 25
    assert len(comparison.models) == 3
    assert len(comparison.baselines) == 3

    reference_summary = comparison.model("logistic-regression-independent")
    svm = comparison.model("linear-svm-independent")
    pls = comparison.model("pls-regression-multioutput")
    assert reference_summary.calibration_ece is not None
    assert svm.calibration_ece is not None
    assert svm.calibration_coefficients
    assert pls.calibration_ece is None
    assert pls.clipping_below_zero_fraction is not None
    assert pls.clipping_above_one_fraction is not None

    paired = svm.paired("mean_balanced_accuracy")
    assert paired.improvement == pytest.approx(
        svm.metrics.mean_balanced_accuracy
        - reference_summary.metrics.mean_balanced_accuracy
    )
    assert paired.lower <= paired.upper
    mse = pls.paired("mean_score_mse")
    assert mse.improvement == pytest.approx(
        reference_summary.metrics.mean_score_mse
        - pls.metrics.mean_score_mse
    )


def test_protocol_comparison_rejects_direction_mismatch(tmp_path: Path) -> None:
    reference = _write_reference(tmp_path, "cross-subject")
    regressor = _write_regressor(tmp_path, "within-subject")

    with pytest.raises(ValueError, match="requires directions"):
        compare_protocol_models(
            "cross-subject",
            reference_run_dirs=reference,
            model_run_dirs={"pls-regression-multioutput": regressor},
            n_resamples=5,
        )
