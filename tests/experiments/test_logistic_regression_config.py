from pathlib import Path

import pytest
from pydantic import ValidationError

from experiments.logistic_regression import (
    LogisticRegressionExperimentConfig,
    build_evaluation_config_hash,
    build_experiment_config_hash,
    load_logistic_regression_config,
    parse_dotted_overrides,
)


def test_loads_default_logistic_regression_config() -> None:
    config = load_logistic_regression_config()

    assert config.dataset.dataset_dir == Path("data/Data_Pattern")
    assert config.dataset.recording_family == "patt"
    assert config.dataset.pattern_type == "random"
    assert (config.dataset.image_rows, config.dataset.image_columns) == (6, 6)
    assert config.split.test_size == 0.2
    assert config.split.group_by == "subject"
    assert config.split.random_state == 42
    assert config.cross_validation.n_splits == 5
    assert config.cross_validation.scoring == "balanced_accuracy"
    assert config.feature_screening.select_k == 100
    assert config.feature_screening.class_weight == "balanced"
    assert config.grid_search.n_jobs == -1
    assert config.prediction_threshold == 0.5
    assert config.artifacts.root == Path("artifacts/experiments/logistic-regression")
    assert config.artifacts.schema_version == 2


def test_config_hash_is_stable_and_versioned() -> None:
    config = load_logistic_regression_config()

    assert build_experiment_config_hash(config) == build_experiment_config_hash(config)
    assert build_experiment_config_hash(config) != build_experiment_config_hash(
        config,
        experiment_version=2,
    )
    assert build_experiment_config_hash(config) != build_experiment_config_hash(
        load_logistic_regression_config(overrides={"random_state": 7})
    )
    assert build_evaluation_config_hash(
        config,
        protocol="within-subject",
        direction="trial-1-to-trial-2",
    ) != build_evaluation_config_hash(
        config,
        protocol="within-subject",
        direction="trial-2-to-trial-1",
    )


def test_parses_dotted_omegaconf_overrides() -> None:
    overrides = parse_dotted_overrides(
        (
            "grid_search.c_values=[0.1,1.0]",
            "grid_search.class_weights=[null,balanced]",
            "grid_search.n_jobs=1",
            "artifacts.overwrite=false",
        )
    )
    config = load_logistic_regression_config(overrides=overrides)

    assert config.grid_search.c_values == (0.1, 1.0)
    assert config.grid_search.class_weights == (None, "balanced")
    assert config.grid_search.n_jobs == 1
    assert config.artifacts.overwrite is False


def test_rejects_invalid_dotted_override_syntax() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_dotted_overrides(("grid_search.n_jobs",))


@pytest.mark.parametrize(
    "overrides",
    [
        {"dataset": {"recording_family": "exec"}},
        {"dataset": {"pattern_type": "geometric"}},
        {"split": {"test_size": 1.0}},
        {"split": {"group_by": "block"}},
        {"cross_validation": {"n_splits": 1}},
        {"feature_screening": {"candidates": []}},
        {"feature_screening": {"c": 0.0}},
        {"grid_search": {"select_k": []}},
        {"grid_search": {"n_jobs": 0}},
        {"prediction_threshold": 0.0},
        {"unexpected": True},
    ],
)
def test_rejects_invalid_experiment_configuration(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        load_logistic_regression_config(overrides=overrides)


def test_rejects_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_logistic_regression_config(config_path=tmp_path / "missing.yaml")


def test_config_model_rejects_unknown_fields() -> None:
    payload = load_logistic_regression_config().model_dump(mode="python")
    payload["unknown"] = 1

    with pytest.raises(ValidationError, match="unknown"):
        LogisticRegressionExperimentConfig.model_validate(payload)
