import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.logistic_regression import (
    load_evaluation_run,
    run_evaluation_protocol,
    summarize_evaluation_runs,
    write_protocol_evaluation_runs,
)
from tests.experiments.test_logistic_regression_runner import (
    _protocol_inputs,
    _runner_config,
)


def _config_with_root(tmp_path: Path) -> object:
    config = _runner_config()
    return config.model_copy(
        update={
            "artifacts": config.artifacts.model_copy(
                update={"root": tmp_path / "runs"}
            )
        }
    )


def test_schema_v1_reference_run_is_evaluated_without_joblib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = Path(
        "artifacts/experiments/logistic-regression/f515948b6bf5af55"
    )

    def reject_joblib(*args: object, **kwargs: object) -> object:
        raise AssertionError("Safe evaluation must not load joblib")

    from experiments.logistic_regression import artifacts as artifacts_module

    monkeypatch.setattr(artifacts_module.joblib, "load", reject_joblib)
    loaded = load_evaluation_run(run_dir)

    assert loaded.manifest["schema_version"] == 1
    assert loaded.evaluation["protocol"] == "cross-subject"
    assert loaded.evaluation["direction"]["name"] == "cross-subject"
    assert loaded.evaluation["split_audit"]["inferred_from_schema_v1"] is True
    assert loaded.evaluation["model_metrics"]["mean_sample_iou"] == pytest.approx(
        0.335257970
    )
    assert loaded.evaluation["model_metrics"]["hamming_loss"] == pytest.approx(
        0.485754986
    )


def test_schema_v2_round_trip_duplicate_and_corruption(tmp_path: Path) -> None:
    dataset, targets = _protocol_inputs()
    config = _config_with_root(tmp_path)
    result = run_evaluation_protocol(
        "cross-subject",
        config=config,
        dataset=dataset,
        targets=targets,
    )
    (run_dir,) = write_protocol_evaluation_runs(
        result,
        targets=targets,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )

    loaded = load_evaluation_run(run_dir)
    assert loaded.manifest["schema_version"] == 2
    assert loaded.evaluation["protocol"] == "cross-subject"
    assert loaded.evaluation["split"]["n_test_rows"] == 18
    assert loaded.evaluation["split_audit"]["has_forbidden_leakage"] is False
    assert loaded.evaluation["selected_feature_family"]
    np.testing.assert_allclose(
        loaded.probabilities,
        result.directions[0].grid_search.probabilities,
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(FileExistsError, match="immutable"):
        write_protocol_evaluation_runs(
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
        load_evaluation_run(run_dir)

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
        load_evaluation_run(run_dir)


def test_complementary_within_subject_runs_are_combined_without_rewrite(
    tmp_path: Path,
) -> None:
    dataset, targets = _protocol_inputs()
    config = _config_with_root(tmp_path)
    result = run_evaluation_protocol(
        "within-subject",
        config=config,
        dataset=dataset,
        targets=targets,
    )
    run_dirs = write_protocol_evaluation_runs(
        result,
        targets=targets,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )
    manifests_before = [
        (run_dir / "manifest.json").read_bytes() for run_dir in run_dirs
    ]

    payload = summarize_evaluation_runs(list(reversed(run_dirs)))

    assert payload["combined"] is not None
    combined = payload["combined"]
    assert combined["direction_names"] == [
        "trial-1-to-trial-2",
        "trial-2-to-trial-1",
    ]
    assert combined["n_test_rows"] == 72
    assert combined["n_subjects"] == 12
    assert result.combined is not None
    assert combined["model_metrics"]["mean_balanced_accuracy"] == pytest.approx(
        result.combined.model_metrics.mean_balanced_accuracy
    )
    assert [
        (run_dir / "manifest.json").read_bytes() for run_dir in run_dirs
    ] == manifests_before


def test_two_non_complementary_runs_are_rejected(tmp_path: Path) -> None:
    dataset, targets = _protocol_inputs()
    config = _config_with_root(tmp_path)
    result = run_evaluation_protocol(
        "cross-subject",
        config=config,
        dataset=dataset,
        targets=targets,
    )
    (run_dir,) = write_protocol_evaluation_runs(
        result,
        targets=targets,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )

    with pytest.raises(ValueError, match="complementary"):
        summarize_evaluation_runs([run_dir, run_dir])


def test_schema_v2_evaluation_json_is_deterministically_formatted(
    tmp_path: Path,
) -> None:
    dataset, targets = _protocol_inputs()
    config = _config_with_root(tmp_path)
    result = run_evaluation_protocol(
        "cross-subject",
        config=config,
        dataset=dataset,
        targets=targets,
    )
    (run_dir,) = write_protocol_evaluation_runs(
        result,
        targets=targets,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )

    text = (run_dir / "evaluation.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    assert text == json.dumps(
        payload,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
