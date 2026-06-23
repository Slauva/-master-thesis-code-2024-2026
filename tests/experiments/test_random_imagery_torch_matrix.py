import json
from pathlib import Path

import pytest

from experiments.random_imagery.config import parse_dotted_overrides
from experiments.random_imagery_torch.config import (
    PRIMARY_TORCH_MODEL_IDS,
    load_torch_config,
)
from experiments.random_imagery_torch.matrix import (
    TORCH_MATRIX_PROTOCOLS,
    TorchMatrixRunSpec,
    build_torch_matrix_plan,
    build_torch_matrix_plan_payload,
    execute_torch_matrix_sweep,
)
from tests.experiments.test_random_imagery_torch_training import (
    _synthetic_inputs,
    _tiny_model_factory,
)


def test_torch_matrix_plan_enumerates_primary_model_protocol_grid() -> None:
    specs = build_torch_matrix_plan()

    assert len(specs) == 12 * 2
    assert sum(spec.expected_direction_runs for spec in specs) == 12 * 3
    assert {spec.model_id for spec in specs} == set(PRIMARY_TORCH_MODEL_IDS)
    assert {spec.protocol for spec in specs} == set(TORCH_MATRIX_PROTOCOLS)
    assert {spec.architecture for spec in specs} == {
        "deep-convnet",
        "eegnet",
        "shallow-convnet",
    }
    assert {spec.method for spec in specs} == {"fft", "morlet", "stft", "superlet"}
    assert len({spec.plan_id for spec in specs}) == len(specs)


def test_torch_matrix_spec_uses_full_dataset_and_full_imagery_root() -> None:
    spec = TorchMatrixRunSpec(
        model_id="eegnet-fft-multilabel",
        protocol="within-subject",
        artifact_root=Path("artifacts/test-full-imagery/torch"),
    )
    overrides = parse_dotted_overrides(spec.dotted_overrides)
    config = load_torch_config(spec.model_id, overrides=overrides)

    assert spec.architecture == "eegnet"
    assert spec.method == "fft"
    assert spec.expected_direction_runs == 2
    assert config.dataset.pattern_type is None
    assert config.dataset.target_sample_types == ("geometric", "random")
    assert config.artifacts.root == Path("artifacts/test-full-imagery/torch")
    assert spec.command[:5] == (
        "random-imagery-torch",
        "run",
        "--model",
        "eegnet-fft-multilabel",
        "--protocol",
    )
    assert "dataset.pattern_type=null" in spec.dotted_overrides


def test_torch_matrix_plan_payload_is_json_ready_and_counts_direction_runs() -> None:
    specs = build_torch_matrix_plan(
        model_ids=("eegnet-fft-multilabel",),
        protocols=("cross-subject", "within-subject"),
        artifact_root=Path("artifacts/test-full-imagery/torch"),
    )
    payload = build_torch_matrix_plan_payload(specs)

    assert payload["run_count"] == 2
    assert payload["expected_direction_run_count"] == 3
    assert payload["architectures"] == ["eegnet"]
    assert payload["methods"] == ["fft"]
    assert json.loads(json.dumps(payload))["runs"][0]["method"] == "fft"


def test_torch_matrix_rejects_shared_dataset_for_mixed_methods() -> None:
    dataset, targets, _ = _synthetic_inputs()
    specs = build_torch_matrix_plan(
        model_ids=("eegnet-fft-multilabel", "eegnet-morlet-multilabel"),
        protocols=("cross-subject",),
        artifact_root=Path("artifacts/test-full-imagery/torch"),
    )

    with pytest.raises(ValueError, match="one preprocessing method"):
        execute_torch_matrix_sweep(
            specs=specs,
            spectral_dataset=dataset,
            targets=targets,
            model_factory=_tiny_model_factory,
        )


def test_torch_matrix_sweep_executes_synthetic_spec_and_writes_summary(
    tmp_path: Path,
) -> None:
    dataset, targets, _ = _synthetic_inputs()
    dataset.config_hash = "synthetic-spectral-config"
    spec = TorchMatrixRunSpec(
        model_id="eegnet-fft-multilabel",
        protocol="cross-subject",
        artifact_root=tmp_path / "runs",
    )

    summary = execute_torch_matrix_sweep(
        specs=(spec,),
        output_path=tmp_path / "summary.json",
        failure_log_path=tmp_path / "failures.json",
        extra_overrides={
            "split": {"test_size": 0.25},
            "training": {
                "batch_size": 4,
                "maximum_epochs": 1,
                "early_stopping_patience": 1,
                "final_seeds": [11, 12, 13],
                "device": "cpu",
            },
            "bootstrap_iterations": 5,
        },
        spectral_dataset=dataset,
        targets=targets,
        model_factory=_tiny_model_factory,
    )

    assert summary["complete"] is True
    assert summary["completed_protocol_run_count"] == 1
    assert summary["completed_direction_run_count"] == 1
    assert summary["failed_protocol_run_count"] == 0
    run = summary["results"][0]
    assert run["status"] == "completed"
    assert run["summary"]["runs"][0]["model_id"] == "eegnet-fft-multilabel"
    assert run["summary"]["runs"][0]["architecture"] == "eegnet"
    assert run["summary"]["runs"][0]["method"] == "fft"
    assert run["summary"]["runs"][0]["score_semantics"] == "native_probability"
    assert Path(run["run_dirs"][0]).is_dir()
    assert json.loads((tmp_path / "summary.json").read_text())["complete"] is True
    assert json.loads((tmp_path / "failures.json").read_text())["failures"] == []
