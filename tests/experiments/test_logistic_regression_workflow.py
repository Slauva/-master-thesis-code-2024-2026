from pathlib import Path

import pytest

from experiments.logistic_regression import (
    execute_evaluation_protocol,
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


def test_workflow_trains_then_reuses_without_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, targets = _protocol_inputs()
    dataset.config_hash = "synthetic-feature-config"
    config = _config_with_root(tmp_path)

    trained = execute_evaluation_protocol(
        "within-subject",
        config=config,
        reuse_existing=True,
        dataset=dataset,
        targets=targets,
    )
    assert trained.reused is False
    assert len(trained.run_dirs) == 2

    from experiments.logistic_regression import workflow as workflow_module

    def reject_fit(*args: object, **kwargs: object) -> object:
        raise AssertionError("Reuse must not invoke the evaluation runner")

    monkeypatch.setattr(workflow_module, "run_evaluation_protocol", reject_fit)
    reused = execute_evaluation_protocol(
        "within-subject",
        config=config,
        reuse_existing=True,
        dataset=dataset,
        targets=targets,
    )

    assert reused.reused is True
    assert reused.run_dirs == trained.run_dirs
    assert reused.summary["combined"] is not None


def test_workflow_rejects_incomplete_reuse_set(
    tmp_path: Path,
) -> None:
    dataset, targets = _protocol_inputs()
    dataset.config_hash = "synthetic-feature-config"
    config = _config_with_root(tmp_path)
    trained = execute_evaluation_protocol(
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
            execute_evaluation_protocol(
                "within-subject",
                config=config,
                reuse_existing=True,
                dataset=dataset,
                targets=targets,
            )
    finally:
        moved_run.rename(second_run)
