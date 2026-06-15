import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.random_imagery_torch.artifacts import (
    build_torch_direction_result,
    load_torch_run,
    replay_torch_predictions,
    write_torch_direction_run,
)
from experiments.random_imagery_torch.config import TorchExperimentConfig
from experiments.random_imagery_torch.training import fit_torch_ensemble, predict_torch_ensemble
from tests.experiments.test_random_imagery_torch_training import (
    _config as _training_config,
)
from tests.experiments.test_random_imagery_torch_training import (
    _synthetic_inputs,
    _tiny_model_factory,
)


def _experiment_config(tmp_path: Path) -> TorchExperimentConfig:
    return TorchExperimentConfig(
        model_id="eegnet-fft-multilabel",
        training=_training_config(maximum_epochs=1, early_stopping_patience=1),
        bootstrap_iterations=5,
        artifacts={"root": tmp_path / "runs", "schema_version": 1},
    )


def _direction_result(tmp_path: Path) -> tuple[object, object, object, object]:
    dataset, targets, direction = _synthetic_inputs()
    config = _experiment_config(tmp_path)
    fitted = fit_torch_ensemble(
        dataset,  # type: ignore[arg-type]
        targets,
        direction,
        config=config.training,
        model_factory=_tiny_model_factory,
    )
    prediction = predict_torch_ensemble(
        fitted,
        dataset,  # type: ignore[arg-type]
        targets,
        direction.test_indices,
        config=config.training,
        model_factory=_tiny_model_factory,
    )
    from experiments.random_imagery.data import audit_evaluation_direction

    result = build_torch_direction_result(
        direction=direction,
        audit=audit_evaluation_direction(targets, direction),
        fitted=fitted,
        prediction=prediction,
        targets=targets,
        config=config,
    )
    return dataset, targets, config, result


def test_torch_artifact_round_trip_safe_load_and_trusted_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, targets, config, result = _direction_result(tmp_path)
    run_dir = write_torch_direction_run(
        result,  # type: ignore[arg-type]
        targets=targets,  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        spectral_config_hash="synthetic-spectral-config",
    )

    from experiments.random_imagery_torch import artifacts as artifacts_module

    original_load = artifacts_module.torch.load

    def reject_torch_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("Safe Torch loading must not deserialize checkpoint weights")

    monkeypatch.setattr(artifacts_module.torch, "load", reject_torch_load)
    safe = load_torch_run(run_dir)
    assert safe.manifest["schema_version"] == 1
    assert safe.checkpoints == ()
    np.testing.assert_allclose(
        safe.scores,
        result.prediction.scores,  # type: ignore[attr-defined]
        rtol=0.0,
        atol=0.0,
    )

    seen_kwargs: list[dict[str, object]] = []

    def recording_load(*args: object, **kwargs: object) -> object:
        seen_kwargs.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(artifacts_module.torch, "load", recording_load)
    trusted = load_torch_run(run_dir, trusted=True)
    assert len(trusted.checkpoints) == 3
    assert seen_kwargs and all(item.get("weights_only") is True for item in seen_kwargs)
    scores, predictions = replay_torch_predictions(
        trusted,
        source_dataset=dataset,  # type: ignore[arg-type]
        targets=targets,  # type: ignore[arg-type]
        model_factory=_tiny_model_factory,
    )
    np.testing.assert_allclose(scores, trusted.scores, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(predictions, trusted.predictions)

    with pytest.raises(FileExistsError, match="immutable"):
        write_torch_direction_run(
            result,  # type: ignore[arg-type]
            targets=targets,  # type: ignore[arg-type]
            config=config,  # type: ignore[arg-type]
            spectral_config_hash="synthetic-spectral-config",
        )


def test_torch_artifact_corruption_metric_and_unsafe_checkpoint_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, targets, config, result = _direction_result(tmp_path)
    run_dir = write_torch_direction_run(
        result,  # type: ignore[arg-type]
        targets=targets,  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        spectral_config_hash="synthetic-spectral-config",
    )
    evaluation_path = run_dir / "evaluation.json"
    manifest_path = run_dir / "manifest.json"
    original_evaluation = evaluation_path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    with evaluation_path.open("a", encoding="utf-8") as file:
        file.write(" ")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        load_torch_run(run_dir)

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
        load_torch_run(run_dir)

    evaluation_path.write_bytes(original_evaluation)
    training_path = run_dir / "training.json"
    training = json.loads(training_path.read_bytes())
    training["ensemble_members"][0]["checkpoint_file"] = "../outside.pt"
    training_path.write_text(
        json.dumps(training, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(original_manifest)
    for path in (evaluation_path, training_path):
        file_content = path.read_bytes()
        manifest["files"][path.name] = {
            "sha256": hashlib.sha256(file_content).hexdigest(),
            "size": len(file_content),
        }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    from experiments.random_imagery_torch import artifacts as artifacts_module

    monkeypatch.setattr(
        artifacts_module.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("Safe loading invoked torch.load"),
    )
    with pytest.raises(ValueError, match="Unsafe Torch checkpoint filename"):
        load_torch_run(run_dir)

