import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal, Protocol, Self

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.random_imagery.registry import get_model_spec


class DatasetSelectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_dir: Path = Path("data/Data_Pattern")
    recording_family: Literal["patt"] = "patt"
    pattern_type: Literal["geometric", "random"] | None = "random"
    image_rows: Literal[6] = 6
    image_columns: Literal[6] = 6
    feature_config_path: Path = Path("confs/features/default.yaml")

    @property
    def target_sample_types(self) -> tuple[Literal["geometric", "random"], ...]:
        if self.pattern_type is None:
            return ("geometric", "random")
        return (self.pattern_type,)


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


class ArtifactConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path = Path("artifacts/experiments/logistic-regression")
    schema_version: Annotated[int, Field(ge=1)] = 2
    overwrite: bool = False


class ClassifierFeatureScreeningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    select_k: Annotated[int, Field(ge=1)] = 100
    variance_threshold: Annotated[float, Field(ge=0.0)] = 0.0
    regularization: Annotated[float, Field(gt=0.0)] = 1.0
    class_weight: Literal["balanced"] | None = "balanced"
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


class LinearSVMGridSearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimator_family: Literal["linear_svm"] = "linear_svm"
    select_k: tuple[Annotated[int, Field(ge=1)], ...]
    c_values: tuple[Annotated[float, Field(gt=0.0)], ...]
    class_weights: tuple[Literal["balanced"] | None, ...]
    max_iter: Annotated[int, Field(ge=1)] = 10_000
    tolerance: Annotated[float, Field(gt=0.0)] = 1e-4
    n_jobs: int = -1
    error_score: Literal["raise"] = "raise"

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        _validate_unique_grid_values(
            select_k=self.select_k,
            regularization=self.c_values,
            class_weights=self.class_weights,
            n_jobs=self.n_jobs,
        )
        return self


class RidgeClassifierGridSearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimator_family: Literal["ridge_classifier"] = "ridge_classifier"
    select_k: tuple[Annotated[int, Field(ge=1)], ...]
    alpha_values: tuple[Annotated[float, Field(gt=0.0)], ...]
    class_weights: tuple[Literal["balanced"] | None, ...]
    tolerance: Annotated[float, Field(gt=0.0)] = 1e-4
    solver: Literal["auto"] = "auto"
    n_jobs: int = -1
    error_score: Literal["raise"] = "raise"

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        _validate_unique_grid_values(
            select_k=self.select_k,
            regularization=self.alpha_values,
            class_weights=self.class_weights,
            n_jobs=self.n_jobs,
        )
        return self


ClassifierGridSearchConfig = Annotated[
    LinearSVMGridSearchConfig | RidgeClassifierGridSearchConfig,
    Field(discriminator="estimator_family"),
]


class RegressionFeatureScreeningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    select_k: Annotated[int, Field(ge=1)] = 100
    variance_threshold: Annotated[float, Field(ge=0.0)] = 0.0
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


class RidgeRegressionGridSearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimator_family: Literal["ridge_regression"] = "ridge_regression"
    select_k: tuple[Annotated[int, Field(ge=1)], ...]
    alpha_values: tuple[Annotated[float, Field(gt=0.0)], ...]
    screening_alpha: Annotated[float, Field(gt=0.0)] = 1.0
    tolerance: Annotated[float, Field(gt=0.0)] = 1e-4
    solver: Literal["auto"] = "auto"
    n_jobs: int = -1
    error_score: Literal["raise"] = "raise"

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        _validate_regression_grid_values(
            select_k=self.select_k,
            parameter_values=(self.alpha_values,),
            n_jobs=self.n_jobs,
        )
        return self


class ElasticNetGridSearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimator_family: Literal["elastic_net"] = "elastic_net"
    select_k: tuple[Annotated[int, Field(ge=1)], ...]
    alpha_values: tuple[Annotated[float, Field(gt=0.0)], ...]
    l1_ratios: tuple[Annotated[float, Field(ge=0.0, le=1.0)], ...]
    screening_alpha: Annotated[float, Field(gt=0.0)] = 0.01
    screening_l1_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    max_iter: Annotated[int, Field(ge=1)] = 10_000
    tolerance: Annotated[float, Field(gt=0.0)] = 1e-4
    selection: Literal["cyclic"] = "cyclic"
    n_jobs: int = -1
    error_score: Literal["raise"] = "raise"

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        _validate_regression_grid_values(
            select_k=self.select_k,
            parameter_values=(self.alpha_values, self.l1_ratios),
            n_jobs=self.n_jobs,
        )
        if 1.0 not in self.l1_ratios:
            raise ValueError("ElasticNet grid must include l1_ratio=1.0 for the Lasso variant")
        return self


class RandomForestRegressionGridSearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimator_family: Literal["random_forest"] = "random_forest"
    select_k: tuple[Annotated[int, Field(ge=1)], ...]
    n_estimators: tuple[Annotated[int, Field(ge=1)], ...]
    max_depth: tuple[Annotated[int, Field(ge=1)] | None, ...]
    min_samples_leaf: tuple[Annotated[int, Field(ge=1)], ...]
    max_features: tuple[Annotated[float, Field(gt=0.0, le=1.0)], ...]
    screening_n_estimators: Annotated[int, Field(ge=1)] = 100
    screening_max_depth: Annotated[int, Field(ge=1)] | None = None
    screening_min_samples_leaf: Annotated[int, Field(ge=1)] = 1
    screening_max_features: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    estimator_n_jobs: Literal[1] = 1
    n_jobs: int = -1
    error_score: Literal["raise"] = "raise"

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        _validate_regression_grid_values(
            select_k=self.select_k,
            parameter_values=(
                self.n_estimators,
                self.max_depth,
                self.min_samples_leaf,
                self.max_features,
            ),
            n_jobs=self.n_jobs,
        )
        return self


class PLSRegressionGridSearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimator_family: Literal["pls_regression"] = "pls_regression"
    select_k: tuple[Annotated[int, Field(ge=1)], ...]
    n_components: tuple[Annotated[int, Field(ge=1)], ...]
    screening_n_components: Annotated[int, Field(ge=1)] = 2
    scale: Literal[False] = False
    max_iter: Annotated[int, Field(ge=1)] = 500
    tolerance: Annotated[float, Field(gt=0.0)] = 1e-6
    n_jobs: int = -1
    error_score: Literal["raise"] = "raise"

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        _validate_regression_grid_values(
            select_k=self.select_k,
            parameter_values=(self.n_components,),
            n_jobs=self.n_jobs,
        )
        return self


RegressionGridSearchConfig = Annotated[
    RidgeRegressionGridSearchConfig
    | ElasticNetGridSearchConfig
    | RandomForestRegressionGridSearchConfig
    | PLSRegressionGridSearchConfig,
    Field(discriminator="estimator_family"),
]


class PlattCalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    c: Annotated[float, Field(gt=0.0)] = 1.0
    max_iter: Annotated[int, Field(ge=1)] = 5000
    tolerance: Annotated[float, Field(gt=0.0)] = 1e-6
    solver: Literal["lbfgs"] = "lbfgs"


class CalibratedClassifierExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Literal[
        "linear-svm-independent",
        "ridge-classifier-independent",
    ]
    dataset: DatasetSelectionConfig
    split: SubjectSplitConfig
    cross_validation: CrossValidationConfig
    feature_screening: ClassifierFeatureScreeningConfig
    grid_search: ClassifierGridSearchConfig
    calibration: PlattCalibrationConfig = PlattCalibrationConfig()
    prediction_threshold: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    bootstrap_iterations: Annotated[int, Field(ge=1)] = 2000
    random_state: int = 42
    artifacts: ArtifactConfig

    @model_validator(mode="after")
    def validate_model_grid(self) -> Self:
        spec = get_model_spec(self.model_id)
        if spec.estimator_family != self.grid_search.estimator_family:
            raise ValueError(
                f"Model {self.model_id!r} requires a "
                f"{spec.estimator_family!r} grid, not "
                f"{self.grid_search.estimator_family!r}"
            )
        return self


class RegressionExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Literal[
        "ridge-regression-independent",
        "ridge-regression-multioutput",
        "elastic-net-independent",
        "elastic-net-multioutput",
        "random-forest-independent",
        "random-forest-multioutput",
        "pls-regression-multioutput",
    ]
    dataset: DatasetSelectionConfig
    split: SubjectSplitConfig
    cross_validation: CrossValidationConfig
    feature_screening: RegressionFeatureScreeningConfig
    grid_search: RegressionGridSearchConfig
    prediction_threshold: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    bootstrap_iterations: Annotated[int, Field(ge=1)] = 2000
    random_state: int = 42
    artifacts: ArtifactConfig

    @model_validator(mode="after")
    def validate_model_grid(self) -> Self:
        spec = get_model_spec(self.model_id)
        if spec.estimator_family != self.grid_search.estimator_family:
            raise ValueError(
                f"Model {self.model_id!r} requires a "
                f"{spec.estimator_family!r} grid, not "
                f"{self.grid_search.estimator_family!r}"
            )
        if spec.task != "regressor":
            raise ValueError(f"Model {self.model_id!r} is not a regressor")
        return self


RandomImageryModelConfig = (
    CalibratedClassifierExperimentConfig | RegressionExperimentConfig
)


class FeatureScreeningConfigLike(Protocol):
    candidates: tuple[tuple[str, ...], ...]


class RandomImageryExperimentConfigLike(Protocol):
    dataset: DatasetSelectionConfig
    split: SubjectSplitConfig
    cross_validation: CrossValidationConfig
    feature_screening: FeatureScreeningConfigLike
    prediction_threshold: float
    bootstrap_iterations: int
    random_state: int
    artifacts: ArtifactConfig


def parse_dotted_overrides(values: Iterable[str]) -> dict[str, Any]:
    dotlist = list(values)
    if any("=" not in value for value in dotlist):
        invalid = next(value for value in dotlist if "=" not in value)
        raise ValueError(f"Configuration override must use KEY=VALUE syntax: {invalid!r}")
    payload = OmegaConf.to_container(
        OmegaConf.from_dotlist(dotlist),
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("Resolved configuration overrides must be a mapping")
    return payload


def load_calibrated_classifier_config(
    model_id: str,
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> CalibratedClassifierExperimentConfig:
    spec = get_model_spec(model_id)
    if spec.estimator_family not in {"linear_svm", "ridge_classifier"}:
        raise ValueError(f"Model {model_id!r} is not a calibrated classifier")
    resolved_path = (
        Path(config_path)
        if config_path is not None
        else _default_classifier_config_path(spec.estimator_family)
    )
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Calibrated classifier configuration does not exist: {resolved_path}"
        )

    configs: list[Any] = [OmegaConf.load(resolved_path)]
    if overrides:
        configs.append(OmegaConf.create(overrides))
    merged = OmegaConf.merge(*configs)
    payload = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved calibrated classifier configuration must be a mapping")
    config = CalibratedClassifierExperimentConfig.model_validate(payload)
    if config.model_id != model_id:
        raise ValueError(
            f"Configuration model {config.model_id!r} does not match requested model {model_id!r}"
        )
    return config


def load_regression_config(
    model_id: str,
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RegressionExperimentConfig:
    spec = get_model_spec(model_id)
    if spec.task != "regressor":
        raise ValueError(f"Model {model_id!r} is not a regressor")
    resolved_path = (
        Path(config_path)
        if config_path is not None
        else _default_regression_config_path(model_id)
    )
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Regression configuration does not exist: {resolved_path}")

    configs: list[Any] = [OmegaConf.load(resolved_path)]
    if overrides:
        configs.append(OmegaConf.create(overrides))
    merged = OmegaConf.merge(*configs)
    payload = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved regression configuration must be a mapping")
    config = RegressionExperimentConfig.model_validate(payload)
    if config.model_id != model_id:
        raise ValueError(
            f"Configuration model {config.model_id!r} does not match requested model {model_id!r}"
        )
    return config


def load_model_config(
    model_id: str,
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RandomImageryModelConfig:
    spec = get_model_spec(model_id)
    if spec.task == "classifier":
        return load_calibrated_classifier_config(
            model_id,
            config_path=config_path,
            overrides=overrides,
        )
    return load_regression_config(
        model_id,
        config_path=config_path,
        overrides=overrides,
    )


def validate_model_config_payload(
    payload: dict[str, Any],
) -> RandomImageryModelConfig:
    model_id = payload.get("model_id")
    if not isinstance(model_id, str):
        raise ValueError("Random-imagery model configuration requires `model_id`")
    spec = get_model_spec(model_id)
    if spec.task == "classifier":
        return CalibratedClassifierExperimentConfig.model_validate(payload)
    return RegressionExperimentConfig.model_validate(payload)


def build_model_run_hash(
    config: RandomImageryModelConfig,
    *,
    protocol: str,
    direction: str,
    experiment_version: int = 3,
    feature_extractor_version: int = 1,
) -> str:
    expected_directions = {
        "cross-subject": {"cross-subject"},
        "within-subject": {"trial-1-to-trial-2", "trial-2-to-trial-1"},
    }
    if protocol not in expected_directions:
        raise ValueError(f"Unsupported evaluation protocol: {protocol!r}")
    if direction not in expected_directions[protocol]:
        raise ValueError(
            f"Direction {direction!r} does not belong to protocol {protocol!r}"
        )
    if experiment_version < 1 or feature_extractor_version < 1:
        raise ValueError("Experiment and feature-extractor versions must be positive")
    payload = {
        "experiment_version": experiment_version,
        "feature_extractor_version": feature_extractor_version,
        "artifact_schema_version": config.artifacts.schema_version,
        "model_id": config.model_id,
        "protocol": protocol,
        "direction": direction,
        "config": config.model_dump(mode="json"),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def _validate_unique_grid_values(
    *,
    select_k: tuple[int, ...],
    regularization: tuple[float, ...],
    class_weights: tuple[str | None, ...],
    n_jobs: int,
) -> None:
    if n_jobs == 0:
        raise ValueError("`n_jobs` must not be zero")
    for name, values in (
        ("select_k", select_k),
        ("regularization", regularization),
        ("class_weights", class_weights),
    ):
        if not values:
            raise ValueError(f"`{name}` must not be empty")
        if len(set(values)) != len(values):
            raise ValueError(f"`{name}` values must be unique")


def _validate_regression_grid_values(
    *,
    select_k: tuple[int, ...],
    parameter_values: tuple[tuple[Any, ...], ...],
    n_jobs: int,
) -> None:
    if n_jobs == 0:
        raise ValueError("`n_jobs` must not be zero")
    values_by_name = (("select_k", select_k),) + tuple(
        (f"parameter_{index}", values)
        for index, values in enumerate(parameter_values)
    )
    for name, values in values_by_name:
        if not values:
            raise ValueError(f"`{name}` must not be empty")
        if len(set(values)) != len(values):
            raise ValueError(f"`{name}` values must be unique")


def _default_classifier_config_path(estimator_family: str) -> Path:
    filename = {
        "linear_svm": "linear_svm.yaml",
        "ridge_classifier": "ridge_classifier.yaml",
    }[estimator_family]
    return Path(__file__).resolve().parents[2] / "confs" / "experiments" / filename


def _default_regression_config_path(model_id: str) -> Path:
    filename = model_id.replace("-", "_") + ".yaml"
    return Path(__file__).resolve().parents[2] / "confs" / "experiments" / filename


__all__ = [
    "ArtifactConfig",
    "CalibratedClassifierExperimentConfig",
    "ClassifierFeatureScreeningConfig",
    "ClassifierGridSearchConfig",
    "CrossValidationConfig",
    "DatasetSelectionConfig",
    "ElasticNetGridSearchConfig",
    "FeatureScreeningConfigLike",
    "LinearSVMGridSearchConfig",
    "PLSRegressionGridSearchConfig",
    "PlattCalibrationConfig",
    "RandomForestRegressionGridSearchConfig",
    "RandomImageryExperimentConfigLike",
    "RandomImageryModelConfig",
    "RegressionExperimentConfig",
    "RegressionFeatureScreeningConfig",
    "RegressionGridSearchConfig",
    "RidgeRegressionGridSearchConfig",
    "RidgeClassifierGridSearchConfig",
    "SubjectSplitConfig",
    "load_calibrated_classifier_config",
    "load_model_config",
    "load_regression_config",
    "build_model_run_hash",
    "parse_dotted_overrides",
    "validate_model_config_payload",
]
