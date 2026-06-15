from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

ModelTopology = Literal["independent", "multioutput"]
ModelTask = Literal["classifier", "regressor"]
ScoreSemantics = Literal[
    "native_probability",
    "calibrated_probability",
    "clipped_regression",
]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    label: str
    estimator_family: str
    topology: ModelTopology
    task: ModelTask
    score_semantics: ScoreSemantics
    reference: bool = False
    exploratory: bool = False

    def __post_init__(self) -> None:
        if not self.model_id or not self.label or not self.estimator_family:
            raise ValueError("Model specifications require non-empty identifiers and labels")
        expected_suffix = f"-{self.topology}"
        if not self.model_id.endswith(expected_suffix):
            raise ValueError(
                f"Model ID {self.model_id!r} must end with topology suffix {expected_suffix!r}"
            )
        expected_semantics = (
            {"native_probability", "calibrated_probability"}
            if self.task == "classifier"
            else {"clipped_regression"}
        )
        if self.score_semantics not in expected_semantics:
            raise ValueError(
                f"Score semantics {self.score_semantics!r} do not match task {self.task!r}"
            )
        if self.reference and self.exploratory:
            raise ValueError("A reference model cannot also be exploratory")


_MODEL_SPECS = (
    ModelSpec(
        model_id="logistic-regression-independent",
        label="Logistic Regression",
        estimator_family="logistic_regression",
        topology="independent",
        task="classifier",
        score_semantics="native_probability",
        reference=True,
    ),
    ModelSpec(
        model_id="linear-svm-independent",
        label="Linear SVM",
        estimator_family="linear_svm",
        topology="independent",
        task="classifier",
        score_semantics="calibrated_probability",
    ),
    ModelSpec(
        model_id="ridge-classifier-independent",
        label="Ridge Classifier",
        estimator_family="ridge_classifier",
        topology="independent",
        task="classifier",
        score_semantics="calibrated_probability",
    ),
    ModelSpec(
        model_id="ridge-regression-independent",
        label="Ridge Regression (independent)",
        estimator_family="ridge_regression",
        topology="independent",
        task="regressor",
        score_semantics="clipped_regression",
    ),
    ModelSpec(
        model_id="ridge-regression-multioutput",
        label="Ridge Regression (multi-output)",
        estimator_family="ridge_regression",
        topology="multioutput",
        task="regressor",
        score_semantics="clipped_regression",
    ),
    ModelSpec(
        model_id="elastic-net-independent",
        label="ElasticNet/Lasso (independent)",
        estimator_family="elastic_net",
        topology="independent",
        task="regressor",
        score_semantics="clipped_regression",
    ),
    ModelSpec(
        model_id="elastic-net-multioutput",
        label="ElasticNet/Lasso (multi-output)",
        estimator_family="elastic_net",
        topology="multioutput",
        task="regressor",
        score_semantics="clipped_regression",
    ),
    ModelSpec(
        model_id="random-forest-independent",
        label="Random Forest Regressor (independent)",
        estimator_family="random_forest",
        topology="independent",
        task="regressor",
        score_semantics="clipped_regression",
    ),
    ModelSpec(
        model_id="random-forest-multioutput",
        label="Random Forest Regressor (multi-output)",
        estimator_family="random_forest",
        topology="multioutput",
        task="regressor",
        score_semantics="clipped_regression",
    ),
    ModelSpec(
        model_id="pls-regression-multioutput",
        label="PLS Regression (multi-output)",
        estimator_family="pls_regression",
        topology="multioutput",
        task="regressor",
        score_semantics="clipped_regression",
        exploratory=True,
    ),
)

MODEL_REGISTRY = MappingProxyType({spec.model_id: spec for spec in _MODEL_SPECS})
if len(MODEL_REGISTRY) != len(_MODEL_SPECS):
    raise RuntimeError("Random-imagery model IDs must be unique")

PLANNED_MODEL_IDS = tuple(spec.model_id for spec in _MODEL_SPECS if not spec.reference)
REFERENCE_MODEL_ID = "logistic-regression-independent"


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as error:
        supported = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unsupported random-imagery model {model_id!r}; expected one of: {supported}"
        ) from error


__all__ = [
    "MODEL_REGISTRY",
    "PLANNED_MODEL_IDS",
    "REFERENCE_MODEL_ID",
    "ModelSpec",
    "ModelTask",
    "ModelTopology",
    "ScoreSemantics",
    "get_model_spec",
]
