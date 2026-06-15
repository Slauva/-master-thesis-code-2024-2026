from pathlib import Path

import pytest

from experiments.random_imagery_torch import execute_torch_protocol
from tests.experiments.test_random_imagery_torch_artifacts import _experiment_config
from tests.experiments.test_random_imagery_torch_training import (
    _synthetic_inputs,
    _tiny_model_factory,
)


def test_torch_workflow_trains_then_reuses_without_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, targets, _ = _synthetic_inputs()
    dataset.config_hash = "synthetic-spectral-config"
    config = _experiment_config(tmp_path)

    trained = execute_torch_protocol(
        "cross-subject",
        config=config,
        reuse_existing=True,
        spectral_dataset=dataset,
        targets=targets,
        model_factory=_tiny_model_factory,
    )
    assert trained.reused is False
    assert len(trained.run_dirs) == 1

    from experiments.random_imagery_torch import workflow as workflow_module

    def reject_fit(*args: object, **kwargs: object) -> object:
        raise AssertionError("Reuse must not invoke Torch fitting")

    monkeypatch.setattr(workflow_module, "fit_torch_ensemble", reject_fit)
    reused = execute_torch_protocol(
        "cross-subject",
        config=config,
        reuse_existing=True,
        spectral_dataset=dataset,
        targets=targets,
        model_factory=_tiny_model_factory,
    )

    assert reused.reused is True
    assert reused.run_dirs == trained.run_dirs

    with pytest.raises(FileExistsError, match="immutable"):
        execute_torch_protocol(
            "cross-subject",
            config=config,
            reuse_existing=False,
            spectral_dataset=dataset,
            targets=targets,
            model_factory=_tiny_model_factory,
        )


def test_torch_workflow_rejects_partial_existing_run_without_reuse(tmp_path: Path) -> None:
    dataset, targets, _ = _synthetic_inputs()
    dataset.config_hash = "synthetic-spectral-config"
    config = _experiment_config(tmp_path)
    execute_torch_protocol(
        "cross-subject",
        config=config,
        reuse_existing=True,
        spectral_dataset=dataset,
        targets=targets,
        model_factory=_tiny_model_factory,
    )
    with pytest.raises(FileExistsError, match="immutable"):
        execute_torch_protocol(
            "cross-subject",
            config=config,
            reuse_existing=False,
            spectral_dataset=dataset,
            targets=targets,
            model_factory=_tiny_model_factory,
        )
