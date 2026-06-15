import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetSelectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_dir: Path = Path("data/Data_Pattern")
    recording_family: Literal["patt"] = "patt"
    pattern_type: Literal["random"] = "random"
    image_rows: Literal[6] = 6
    image_columns: Literal[6] = 6
    feature_config_path: Path = Path("confs/features/default.yaml")


class SubjectSplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    test_size: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.2
    random_state: int = 42
    group_by: Literal["subject"] = "subject"
    require_both_classes: bool = True


class CrossValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_splits: Annotated[int, Field(ge=2)] = 5
    shuffle: Literal[True] = True
    random_state: int = 42
    scoring: Literal["balanced_accuracy"] = "balanced_accuracy"


class FeatureScreeningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    select_k: Annotated[int, Field(ge=1)] = 100
    variance_threshold: Annotated[float, Field(ge=0.0)] = 0.0
    c: Annotated[float, Field(gt=0.0)] = 1.0
    penalty: Literal["l2"] = "l2"
    class_weight: Literal["balanced"] | None = "balanced"
    solver: Literal["liblinear"] = "liblinear"
    max_iter: Annotated[int, Field(ge=1)] = 5000
    candidates: tuple[tuple[str, ...], ...]

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        if not self.candidates:
            raise ValueError("At least one feature-screening candidate is required")
        if any(not candidate for candidate in self.candidates):
            raise ValueError("Feature-screening candidates must not be empty")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Feature-screening candidates must be unique")
        return self


class GridSearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    select_k: tuple[Annotated[int, Field(ge=1)], ...]
    c_values: tuple[Annotated[float, Field(gt=0.0)], ...]
    penalties: tuple[Literal["l1", "l2"], ...]
    class_weights: tuple[Literal["balanced"] | None, ...]
    solver: Literal["liblinear"] = "liblinear"
    max_iter: Annotated[int, Field(ge=1)] = 5000
    n_jobs: int = -1
    error_score: Literal["raise"] = "raise"

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        if self.n_jobs == 0:
            raise ValueError("`n_jobs` must not be zero")
        for name, values in (
            ("select_k", self.select_k),
            ("c_values", self.c_values),
            ("penalties", self.penalties),
            ("class_weights", self.class_weights),
        ):
            if not values:
                raise ValueError(f"`{name}` must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"`{name}` values must be unique")
        return self


class ArtifactConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path = Path("artifacts/experiments/logistic-regression")
    schema_version: Annotated[int, Field(ge=1)] = 1
    overwrite: bool = False


class LogisticRegressionExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: DatasetSelectionConfig
    split: SubjectSplitConfig
    cross_validation: CrossValidationConfig
    feature_screening: FeatureScreeningConfig
    grid_search: GridSearchConfig
    prediction_threshold: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    bootstrap_iterations: Annotated[int, Field(ge=1)] = 2000
    random_state: int = 42
    artifacts: ArtifactConfig


def load_logistic_regression_config(
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> LogisticRegressionExperimentConfig:
    resolved_path = Path(config_path) if config_path is not None else _default_config_path()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Logistic Regression configuration does not exist: {resolved_path}")

    configs: list[Any] = [OmegaConf.load(resolved_path)]
    if overrides:
        configs.append(OmegaConf.create(overrides))
    merged = OmegaConf.merge(*configs)
    payload = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved Logistic Regression configuration must be a mapping")
    return LogisticRegressionExperimentConfig.model_validate(payload)


def build_experiment_config_hash(
    config: LogisticRegressionExperimentConfig,
    *,
    experiment_version: int = 1,
    feature_extractor_version: int = 1,
) -> str:
    if experiment_version < 1 or feature_extractor_version < 1:
        raise ValueError("Experiment and feature-extractor versions must be positive")
    payload = {
        "experiment_version": experiment_version,
        "feature_extractor_version": feature_extractor_version,
        "artifact_schema_version": config.artifacts.schema_version,
        "config": config.model_dump(mode="json"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "confs" / "experiments" / "logistic_regression.yaml"
