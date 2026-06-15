import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.logistic_regression import (
    run_evaluation_protocol as run_logistic_protocol,
)
from experiments.logistic_regression import write_protocol_evaluation_runs
from experiments.random_imagery import (
    LinearSVMBackend,
    PLSRegressionMultiOutputBackend,
    RidgeRegressionIndependentBackend,
    build_aligned_feature_partition,
    compare_runs,
    load_calibrated_classifier_config,
    load_model_run,
    load_regression_config,
    replay_model_predictions,
    run_model_evaluation_protocol,
    summarize_model_runs,
    write_model_protocol_runs,
)
from tests.experiments.test_logistic_regression_runner import (
    _protocol_inputs,
    _runner_config,
)


def _classifier_config(tmp_path: Path) -> object:
    return load_calibrated_classifier_config(
        "linear-svm-independent",
        overrides={
            "split": {"test_size": 0.25},
            "cross_validation": {"n_splits": 3},
            "feature_screening": {
                "select_k": 2,
                "candidates": [["time"]],
            },
            "grid_search": {
                "select_k": [2],
                "c_values": [1.0],
                "class_weights": ["balanced"],
                "n_jobs": 1,
            },
            "bootstrap_iterations": 20,
            "artifacts": {"root": str(tmp_path / "runs")},
        },
    )


def _regression_config(
    tmp_path: Path,
    *,
    multioutput: bool,
) -> object:
    model_id = (
        "pls-regression-multioutput"
        if multioutput
        else "ridge-regression-independent"
    )
    grid = {"n_components": [1]} if multioutput else {"alpha_values": [1.0]}
    return load_regression_config(
        model_id,
        overrides={
            "split": {"test_size": 0.25},
            "cross_validation": {"n_splits": 3},
            "feature_screening": {
                "select_k": 2,
                "candidates": [["time"]],
            },
            "grid_search": {"select_k": [2], **grid, "n_jobs": 1},
            "bootstrap_iterations": 20,
            "artifacts": {"root": str(tmp_path / "runs")},
        },
    )


@pytest.mark.parametrize(
    ("config_factory", "backend", "expected_pipelines"),
    [
        (_classifier_config, LinearSVMBackend(), 2),
        (
            lambda path: _regression_config(path, multioutput=False),
            RidgeRegressionIndependentBackend(),
            2,
        ),
        (
            lambda path: _regression_config(path, multioutput=True),
            PLSRegressionMultiOutputBackend(),
            1,
        ),
    ],
)
def test_schema_v3_round_trip_safe_load_and_trusted_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_factory: object,
    backend: object,
    expected_pipelines: int,
) -> None:
    dataset, targets = _protocol_inputs()
    config = config_factory(tmp_path)
    result = run_model_evaluation_protocol(
        "cross-subject",
        config=config,
        backend=backend,
        dataset=dataset,
        targets=targets,
    )
    (run_dir,) = write_model_protocol_runs(
        result,
        targets=targets,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )

    from experiments.random_imagery import artifacts as artifacts_module

    original_load = artifacts_module.joblib.load

    def reject_joblib(*args: object, **kwargs: object) -> object:
        raise AssertionError("Safe schema-v3 loading must not deserialize joblib")

    monkeypatch.setattr(artifacts_module.joblib, "load", reject_joblib)
    safe = load_model_run(run_dir)
    assert safe.manifest["schema_version"] == 3
    assert safe.pipelines == ()
    assert safe.results["pipeline_count"] == expected_pipelines
    np.testing.assert_allclose(
        safe.scores,
        result.directions[0].prediction.scores,
        rtol=0.0,
        atol=0.0,
    )

    monkeypatch.setattr(artifacts_module.joblib, "load", original_load)
    trusted = load_model_run(run_dir, trusted=True)
    assert len(trusted.pipelines) == expected_pipelines
    test_features = build_aligned_feature_partition(
        dataset,
        targets=targets,
        row_indices=result.directions[0].direction.test_indices,
        block_names=result.directions[0].fitted_model.selected_block_names,
    )
    scores, predictions = replay_model_predictions(
        trusted,
        test_features=test_features,
    )
    np.testing.assert_allclose(scores, trusted.scores, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(predictions, trusted.predictions)

    with pytest.raises(FileExistsError, match="immutable"):
        write_model_protocol_runs(
            result,
            targets=targets,
            config=config,
            feature_config_hash="synthetic-feature-config",
        )


def test_schema_v3_corruption_and_metric_tampering_are_rejected(
    tmp_path: Path,
) -> None:
    dataset, targets = _protocol_inputs()
    config = _regression_config(tmp_path, multioutput=True)
    result = run_model_evaluation_protocol(
        "cross-subject",
        config=config,
        backend=PLSRegressionMultiOutputBackend(),
        dataset=dataset,
        targets=targets,
    )
    (run_dir,) = write_model_protocol_runs(
        result,
        targets=targets,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )
    evaluation_path = run_dir / "evaluation.json"
    manifest_path = run_dir / "manifest.json"
    original_evaluation = evaluation_path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    with evaluation_path.open("a", encoding="utf-8") as file:
        file.write(" ")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        load_model_run(run_dir)

    evaluation_path.write_bytes(original_evaluation)
    manifest_path.write_bytes(original_manifest)
    evaluation = json.loads(original_evaluation)
    evaluation["model_metrics"]["mean_sample_iou"] = -1.0
    evaluation_path.write_text(
        json.dumps(evaluation, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(original_manifest)
    content = evaluation_path.read_bytes()
    manifest["files"]["evaluation.json"] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="metric differs from arrays"):
        load_model_run(run_dir)


def test_schema_v3_safe_load_rejects_unsafe_pipeline_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, targets = _protocol_inputs()
    config = _regression_config(tmp_path, multioutput=True)
    result = run_model_evaluation_protocol(
        "cross-subject",
        config=config,
        backend=PLSRegressionMultiOutputBackend(),
        dataset=dataset,
        targets=targets,
    )
    (run_dir,) = write_model_protocol_runs(
        result,
        targets=targets,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )
    results_path = run_dir / "results.json"
    manifest_path = run_dir / "manifest.json"
    results = json.loads(results_path.read_bytes())
    results["pipeline_files"] = ["../outside.joblib"]
    results_path.write_text(
        json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_bytes())
    content = results_path.read_bytes()
    manifest["files"]["results.json"] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    from experiments.random_imagery import artifacts as artifacts_module

    monkeypatch.setattr(
        artifacts_module.joblib,
        "load",
        lambda *args, **kwargs: pytest.fail("Safe loading invoked joblib.load"),
    )
    with pytest.raises(ValueError, match="Unsafe experiment pipeline filename"):
        load_model_run(run_dir)


def test_schema_v3_within_subject_runs_combine_without_rewrite(
    tmp_path: Path,
) -> None:
    dataset, targets = _protocol_inputs()
    config = _regression_config(tmp_path, multioutput=True)
    result = run_model_evaluation_protocol(
        "within-subject",
        config=config,
        backend=PLSRegressionMultiOutputBackend(),
        dataset=dataset,
        targets=targets,
    )
    run_dirs = write_model_protocol_runs(
        result,
        targets=targets,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )
    manifests = [(run_dir / "manifest.json").read_bytes() for run_dir in run_dirs]

    summary = summarize_model_runs(list(reversed(run_dirs)))

    assert summary["combined"] is not None
    assert summary["combined"]["n_test_rows"] == 72
    assert result.combined is not None
    assert summary["combined"]["model_metrics"]["mean_balanced_accuracy"] == pytest.approx(
        result.combined.model_metrics.mean_balanced_accuracy
    )
    assert [(run_dir / "manifest.json").read_bytes() for run_dir in run_dirs] == manifests


def test_mixed_schema_v2_v3_comparison_uses_identical_split(
    tmp_path: Path,
) -> None:
    logistic_dataset, logistic_targets = _protocol_inputs()
    logistic_config = _runner_config().model_copy(
        update={
            "artifacts": _runner_config().artifacts.model_copy(
                update={"root": tmp_path / "logistic"}
            )
        }
    )
    logistic_result = run_logistic_protocol(
        "cross-subject",
        config=logistic_config,
        dataset=logistic_dataset,
        targets=logistic_targets,
    )
    (schema_v2_dir,) = write_protocol_evaluation_runs(
        logistic_result,
        targets=logistic_targets,
        config=logistic_config,
        feature_config_hash="synthetic-feature-config",
    )

    model_dataset, model_targets = _protocol_inputs()
    model_config = _regression_config(tmp_path, multioutput=True)
    model_result = run_model_evaluation_protocol(
        "cross-subject",
        config=model_config,
        backend=PLSRegressionMultiOutputBackend(),
        dataset=model_dataset,
        targets=model_targets,
    )
    (schema_v3_dir,) = write_model_protocol_runs(
        model_result,
        targets=model_targets,
        config=model_config,
        feature_config_hash="synthetic-feature-config",
    )

    comparison = compare_runs([schema_v2_dir, schema_v3_dir])

    assert comparison["protocol"] == "cross-subject"
    assert comparison["n_test_rows"] == 18
    assert [run["artifact_schema_version"] for run in comparison["runs"]] == [2, 3]
    assert [run["model_id"] for run in comparison["runs"]] == [
        "logistic-regression-independent",
        "pls-regression-multioutput",
    ]
