from pathlib import Path

import pytest

from experiments.random_imagery import (
    execute_model_protocol,
    load_regression_config,
)
from tests.experiments.test_logistic_regression_runner import _protocol_inputs


def _config(tmp_path: Path) -> object:
    return load_regression_config(
        "pls-regression-multioutput",
        overrides={
            "cross_validation": {"n_splits": 3},
            "feature_screening": {
                "select_k": 2,
                "candidates": [["time"]],
            },
            "grid_search": {
                "select_k": [2],
                "n_components": [1],
                "n_jobs": 1,
            },
            "bootstrap_iterations": 20,
            "artifacts": {"root": str(tmp_path / "runs")},
        },
    )


def test_model_workflow_trains_then_reuses_without_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, targets = _protocol_inputs()
    dataset.config_hash = "synthetic-feature-config"
    config = _config(tmp_path)
    trained = execute_model_protocol(
        "within-subject",
        config=config,
        reuse_existing=True,
        dataset=dataset,
        targets=targets,
    )
    assert trained.reused is False
    assert len(trained.run_dirs) == 2

    from experiments.random_imagery import workflow as workflow_module

    def reject_fit(*args: object, **kwargs: object) -> object:
        raise AssertionError("Reuse must not invoke model fitting")

    monkeypatch.setattr(
        workflow_module,
        "run_model_evaluation_protocol",
        reject_fit,
    )
    reused = execute_model_protocol(
        "within-subject",
        config=config,
        reuse_existing=True,
        dataset=dataset,
        targets=targets,
    )

    assert reused.reused is True
    assert reused.run_dirs == trained.run_dirs
    assert reused.summary["combined"] is not None


def test_model_workflow_rejects_incomplete_reuse_set(tmp_path: Path) -> None:
    dataset, targets = _protocol_inputs()
    dataset.config_hash = "synthetic-feature-config"
    config = _config(tmp_path)
    trained = execute_model_protocol(
        "within-subject",
        config=config,
        reuse_existing=True,
        dataset=dataset,
        targets=targets,
    )
    second_run = trained.run_dirs[1]
    moved_run = second_run.with_name(f".{second_run.name}-held")
    second_run.rename(moved_run)
    try:
        with pytest.raises(FileNotFoundError, match="incomplete"):
            execute_model_protocol(
                "within-subject",
                config=config,
                reuse_existing=True,
                dataset=dataset,
                targets=targets,
            )
    finally:
        moved_run.rename(second_run)
