import json
from pathlib import Path

import pytest

from experiments.logistic_regression.config import load_logistic_regression_config
from experiments.random_imagery.config import load_model_config, parse_dotted_overrides
from experiments.random_imagery.matrix import (
    CLASSICAL_MATRIX_MODEL_IDS,
    MATRIX_PROTOCOLS,
    TABULAR_FEATURE_FAMILIES,
    MatrixRunSpec,
    build_classical_matrix_plan,
    build_matrix_plan_payload,
    execute_classical_matrix_sweep,
    feature_family_from_slug,
    feature_family_slug,
)
from tests.experiments.test_logistic_regression_runner import _protocol_inputs


def test_classical_matrix_plan_enumerates_full_model_feature_protocol_grid() -> None:
    specs = build_classical_matrix_plan()
    assert len(specs) == 10 * 9 * 2
    assert sum(spec.expected_direction_runs for spec in specs) == 10 * 9 * 3
    assert {spec.model_id for spec in specs} == set(CLASSICAL_MATRIX_MODEL_IDS)
    assert {spec.feature_family for spec in specs} == set(TABULAR_FEATURE_FAMILIES)
    assert {spec.protocol for spec in specs} == set(MATRIX_PROTOCOLS)
    assert len({spec.plan_id for spec in specs}) == len(specs)


def test_matrix_plan_uses_logistic_reference_and_schema_v3_model_runners() -> None:
    logistic = MatrixRunSpec(
        model_id="logistic-regression-independent",
        feature_family=("time", "spectral"),
        protocol="cross-subject",
    )
    ridge = MatrixRunSpec(
        model_id="ridge-regression-independent",
        feature_family=("lbp",),
        protocol="within-subject",
    )

    assert logistic.runner == "logistic-regression"
    assert logistic.command[:3] == ("logistic-regression", "run", "--protocol")
    assert "--model" not in logistic.command
    assert "dataset.pattern_type=null" in logistic.dotted_overrides
    assert "feature_screening.candidates=[[time,spectral]]" in logistic.dotted_overrides

    assert ridge.runner == "random-imagery-models"
    assert ridge.command[:5] == (
        "random-imagery-models",
        "run",
        "--model",
        "ridge-regression-independent",
        "--protocol",
    )
    assert ridge.expected_direction_runs == 2
    assert ridge.overrides["dataset"]["pattern_type"] is None
    assert ridge.overrides["feature_screening"]["candidates"] == [["lbp"]]


@pytest.mark.parametrize("model_id", CLASSICAL_MATRIX_MODEL_IDS)
def test_matrix_overrides_load_as_single_fixed_full_dataset_candidate(model_id: str) -> None:
    spec = MatrixRunSpec(
        model_id=model_id,
        feature_family=("covariance",),
        protocol="cross-subject",
        artifact_root=Path("artifacts/test-full-imagery"),
    )
    overrides = parse_dotted_overrides(spec.dotted_overrides)

    if model_id == "logistic-regression-independent":
        config = load_logistic_regression_config(overrides=overrides)
    else:
        config = load_model_config(model_id, overrides=overrides)

    assert config.dataset.pattern_type is None
    assert config.dataset.target_sample_types == ("geometric", "random")
    assert config.feature_screening.candidates == (("covariance",),)
    assert config.artifacts.root == Path(
        f"artifacts/test-full-imagery/{model_id}/covariance"
    )


def test_matrix_plan_payload_is_json_ready_and_counts_direction_runs() -> None:
    specs = build_classical_matrix_plan(
        model_ids=("logistic-regression-independent",),
        feature_families=(("time",),),
        protocols=("cross-subject", "within-subject"),
        artifact_root=Path("artifacts/test-full-imagery"),
    )
    payload = build_matrix_plan_payload(specs)

    assert payload["run_count"] == 2
    assert payload["expected_direction_run_count"] == 3
    assert json.loads(json.dumps(payload))["runs"][0]["feature_family"] == ["time"]


def test_feature_family_slug_roundtrip_and_validation() -> None:
    for feature_family in TABULAR_FEATURE_FAMILIES:
        assert feature_family_from_slug(feature_family_slug(feature_family)) == feature_family

    with pytest.raises(ValueError, match="Unsupported feature family"):
        feature_family_from_slug("raw")


def test_matrix_sweep_executes_logistic_reference_spec_and_writes_summary(
    tmp_path: Path,
) -> None:
    dataset, targets = _protocol_inputs()
    dataset.config_hash = "synthetic-feature-config"
    spec = MatrixRunSpec(
        model_id="logistic-regression-independent",
        feature_family=("time",),
        protocol="cross-subject",
        artifact_root=tmp_path / "runs",
    )

    summary = execute_classical_matrix_sweep(
        specs=(spec,),
        output_path=tmp_path / "summary.json",
        failure_log_path=tmp_path / "failures.json",
        extra_overrides={
            "split": {"test_size": 0.25},
            "cross_validation": {"n_splits": 3},
            "feature_screening": {"select_k": 2, "max_iter": 1000},
            "grid_search": {
                "select_k": [2],
                "c_values": [1.0],
                "penalties": ["l2"],
                "class_weights": ["balanced"],
                "max_iter": 1000,
                "n_jobs": 1,
            },
            "bootstrap_iterations": 10,
        },
        dataset=dataset,
        targets=targets,
    )

    assert summary["complete"] is True
    assert summary["completed_protocol_run_count"] == 1
    assert summary["completed_direction_run_count"] == 1
    assert summary["failed_protocol_run_count"] == 0
    run = summary["results"][0]
    assert run["status"] == "completed"
    assert run["summary"]["runs"][0]["model_id"] == "logistic-regression-independent"
    assert run["summary"]["runs"][0]["selected_feature_family"] == ["time"]
    assert Path(run["run_dirs"][0]).is_dir()
    assert json.loads((tmp_path / "summary.json").read_text())["complete"] is True
    assert json.loads((tmp_path / "failures.json").read_text())["failures"] == []


def test_matrix_sweep_executes_schema_v3_model_spec(
    tmp_path: Path,
) -> None:
    dataset, targets = _protocol_inputs()
    dataset.config_hash = "synthetic-feature-config"
    spec = MatrixRunSpec(
        model_id="pls-regression-multioutput",
        feature_family=("spectral",),
        protocol="cross-subject",
        artifact_root=tmp_path / "runs",
    )

    summary = execute_classical_matrix_sweep(
        specs=(spec,),
        output_path=tmp_path / "summary.json",
        failure_log_path=tmp_path / "failures.json",
        extra_overrides={
            "split": {"test_size": 0.25},
            "cross_validation": {"n_splits": 3},
            "feature_screening": {"select_k": 2},
            "grid_search": {
                "select_k": [2],
                "n_components": [1],
                "n_jobs": 1,
            },
            "bootstrap_iterations": 10,
        },
        dataset=dataset,
        targets=targets,
    )

    assert summary["completed_protocol_run_count"] == 1
    run = summary["results"][0]
    assert run["summary"]["runs"][0]["model_id"] == "pls-regression-multioutput"
    assert run["summary"]["runs"][0]["selected_feature_family"] == ["spectral"]
    assert run["summary"]["runs"][0]["score_semantics"] == "clipped_regression"
