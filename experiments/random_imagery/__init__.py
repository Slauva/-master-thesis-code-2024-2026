from importlib import import_module
from typing import Any

from experiments.random_imagery.config import (
    ArtifactConfig,
    CalibratedClassifierExperimentConfig,
    ClassifierFeatureScreeningConfig,
    ClassifierGridSearchConfig,
    CrossValidationConfig,
    DatasetSelectionConfig,
    ElasticNetGridSearchConfig,
    LinearSVMGridSearchConfig,
    PlattCalibrationConfig,
    PLSRegressionGridSearchConfig,
    RandomForestRegressionGridSearchConfig,
    RandomImageryExperimentConfigLike,
    RandomImageryModelConfig,
    RegressionExperimentConfig,
    RegressionFeatureScreeningConfig,
    RegressionGridSearchConfig,
    RidgeClassifierGridSearchConfig,
    RidgeRegressionGridSearchConfig,
    SubjectSplitConfig,
    build_model_run_hash,
    load_calibrated_classifier_config,
    load_model_config,
    load_regression_config,
    parse_dotted_overrides,
)
from experiments.random_imagery.registry import (
    MODEL_REGISTRY,
    PLANNED_MODEL_IDS,
    REFERENCE_MODEL_ID,
    ModelSpec,
    get_model_spec,
)

_LAZY_EXPORTS = {
    "BaselineEvaluation": ("experiments.random_imagery.runner", "BaselineEvaluation"),
    "BaselineProtocolSummary": (
        "experiments.random_imagery.comparison",
        "BaselineProtocolSummary",
    ),
    "CalibrationBin": (
        "experiments.random_imagery.comparison",
        "CalibrationBin",
    ),
    "ClassifierCandidateScore": (
        "experiments.random_imagery.classifier_backend",
        "ClassifierCandidateScore",
    ),
    "ClassifierHyperparameters": (
        "experiments.random_imagery.classifier_backend",
        "ClassifierHyperparameters",
    ),
    "ComparisonRun": (
        "experiments.random_imagery.artifacts",
        "ComparisonRun",
    ),
    "ModelProtocolSummary": (
        "experiments.random_imagery.comparison",
        "ModelProtocolSummary",
    ),
    "CombinedProtocolEvaluation": (
        "experiments.random_imagery.runner",
        "CombinedProtocolEvaluation",
    ),
    "DirectionEvaluationResult": (
        "experiments.random_imagery.runner",
        "DirectionEvaluationResult",
    ),
    "ElasticNetIndependentBackend": (
        "experiments.random_imagery.regression_backend",
        "ElasticNetIndependentBackend",
    ),
    "ElasticNetMultiOutputBackend": (
        "experiments.random_imagery.regression_backend",
        "ElasticNetMultiOutputBackend",
    ),
    "EvaluationDirection": ("experiments.random_imagery.shared", "EvaluationDirection"),
    "EvaluationProtocol": ("experiments.random_imagery.shared", "EvaluationProtocol"),
    "EvaluationProtocolDefinition": (
        "experiments.random_imagery.shared",
        "EvaluationProtocolDefinition",
    ),
    "FittedDirectionModel": (
        "experiments.random_imagery.contracts",
        "FittedDirectionModel",
    ),
    "FittedCalibratedPixelModel": (
        "experiments.random_imagery.classifier_backend",
        "FittedCalibratedPixelModel",
    ),
    "FittedCalibratedPixelModels": (
        "experiments.random_imagery.classifier_backend",
        "FittedCalibratedPixelModels",
    ),
    "FittedIndependentRegressionModel": (
        "experiments.random_imagery.regression_backend",
        "FittedIndependentRegressionModel",
    ),
    "FittedIndependentRegressionModels": (
        "experiments.random_imagery.regression_backend",
        "FittedIndependentRegressionModels",
    ),
    "FittedMultiOutputRegressionModel": (
        "experiments.random_imagery.regression_backend",
        "FittedMultiOutputRegressionModel",
    ),
    "LinearSVMBackend": (
        "experiments.random_imagery.classifier_backend",
        "LinearSVMBackend",
    ),
    "LoadedModelRun": (
        "experiments.random_imagery.artifacts",
        "LoadedModelRun",
    ),
    "MatrixRunSpec": (
        "experiments.random_imagery.matrix",
        "MatrixRunSpec",
    ),
    "LogisticRegressionBackend": (
        "experiments.random_imagery.logistic_backend",
        "LogisticRegressionBackend",
    ),
    "ModelPrediction": ("experiments.random_imagery.contracts", "ModelPrediction"),
    "ModelProtocolWorkflowResult": (
        "experiments.random_imagery.workflow",
        "ModelProtocolWorkflowResult",
    ),
    "MultiTargetSelectKBest": (
        "experiments.random_imagery.regression_backend",
        "MultiTargetSelectKBest",
    ),
    "PLSRegressionMultiOutputBackend": (
        "experiments.random_imagery.regression_backend",
        "PLSRegressionMultiOutputBackend",
    ),
    "PixelTargetDataset": ("experiments.random_imagery.shared", "PixelTargetDataset"),
    "PairedMetricDifference": (
        "experiments.random_imagery.comparison",
        "PairedMetricDifference",
    ),
    "PlattCalibration": (
        "experiments.random_imagery.classifier_backend",
        "PlattCalibration",
    ),
    "PredictionMetrics": ("experiments.random_imagery.shared", "PredictionMetrics"),
    "ProtocolEvaluationResult": (
        "experiments.random_imagery.runner",
        "ProtocolEvaluationResult",
    ),
    "ProtocolComparison": (
        "experiments.random_imagery.comparison",
        "ProtocolComparison",
    ),
    "ProtocolLeakageAudit": (
        "experiments.random_imagery.shared",
        "ProtocolLeakageAudit",
    ),
    "RandomImageryModelBackend": (
        "experiments.random_imagery.contracts",
        "RandomImageryModelBackend",
    ),
    "RandomForestIndependentBackend": (
        "experiments.random_imagery.regression_backend",
        "RandomForestIndependentBackend",
    ),
    "RandomForestMultiOutputBackend": (
        "experiments.random_imagery.regression_backend",
        "RandomForestMultiOutputBackend",
    ),
    "RegressionCandidateScore": (
        "experiments.random_imagery.regression_backend",
        "RegressionCandidateScore",
    ),
    "RegressionFeatureScreeningResult": (
        "experiments.random_imagery.regression_backend",
        "RegressionFeatureScreeningResult",
    ),
    "RidgeRegressionIndependentBackend": (
        "experiments.random_imagery.regression_backend",
        "RidgeRegressionIndependentBackend",
    ),
    "RidgeRegressionMultiOutputBackend": (
        "experiments.random_imagery.regression_backend",
        "RidgeRegressionMultiOutputBackend",
    ),
    "RidgeClassifierBackend": (
        "experiments.random_imagery.classifier_backend",
        "RidgeClassifierBackend",
    ),
    "ScoreDiagnostics": ("experiments.random_imagery.contracts", "ScoreDiagnostics"),
    "SubjectBootstrapInterval": (
        "experiments.random_imagery.shared",
        "SubjectBootstrapInterval",
    ),
    "build_evaluation_protocol": (
        "experiments.random_imagery.shared",
        "build_evaluation_protocol",
    ),
    "build_aligned_feature_partition": (
        "experiments.random_imagery.shared",
        "build_aligned_feature_partition",
    ),
    "build_model_backend": (
        "experiments.random_imagery.backends",
        "build_model_backend",
    ),
    "build_classical_matrix_plan": (
        "experiments.random_imagery.matrix",
        "build_classical_matrix_plan",
    ),
    "build_matrix_plan_payload": (
        "experiments.random_imagery.matrix",
        "build_matrix_plan_payload",
    ),
    "execute_classical_matrix_sweep": (
        "experiments.random_imagery.matrix",
        "execute_classical_matrix_sweep",
    ),
    "build_random_imagery_targets": (
        "experiments.random_imagery.shared",
        "build_random_imagery_targets",
    ),
    "evaluate_prediction_matrix": (
        "experiments.random_imagery.shared",
        "evaluate_prediction_matrix",
    ),
    "compare_runs": (
        "experiments.random_imagery.artifacts",
        "compare_runs",
    ),
    "compare_protocol_models": (
        "experiments.random_imagery.comparison",
        "compare_protocol_models",
    ),
    "execute_model_protocol": (
        "experiments.random_imagery.workflow",
        "execute_model_protocol",
    ),
    "fit_calibrated_pixel_classifiers": (
        "experiments.random_imagery.classifier_backend",
        "fit_calibrated_pixel_classifiers",
    ),
    "fit_independent_regression_models": (
        "experiments.random_imagery.regression_backend",
        "fit_independent_regression_models",
    ),
    "fit_multioutput_regression_model": (
        "experiments.random_imagery.regression_backend",
        "fit_multioutput_regression_model",
    ),
    "load_model_run": (
        "experiments.random_imagery.artifacts",
        "load_model_run",
    ),
    "model_run_dir": (
        "experiments.random_imagery.artifacts",
        "model_run_dir",
    ),
    "replay_model_predictions": (
        "experiments.random_imagery.artifacts",
        "replay_model_predictions",
    ),
    "run_model_evaluation_protocol": (
        "experiments.random_imagery.runner",
        "run_model_evaluation_protocol",
    ),
    "screen_classifier_feature_families": (
        "experiments.random_imagery.classifier_backend",
        "screen_classifier_feature_families",
    ),
    "screen_regression_feature_families": (
        "experiments.random_imagery.regression_backend",
        "screen_regression_feature_families",
    ),
    "summarize_model_runs": (
        "experiments.random_imagery.artifacts",
        "summarize_model_runs",
    ),
    "write_model_protocol_runs": (
        "experiments.random_imagery.artifacts",
        "write_model_protocol_runs",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "MODEL_REGISTRY",
    "PLANNED_MODEL_IDS",
    "REFERENCE_MODEL_ID",
    "ArtifactConfig",
    "BaselineEvaluation",
    "CalibratedClassifierExperimentConfig",
    "ClassifierCandidateScore",
    "ClassifierFeatureScreeningConfig",
    "ClassifierGridSearchConfig",
    "ClassifierHyperparameters",
    "ComparisonRun",
    "CombinedProtocolEvaluation",
    "CrossValidationConfig",
    "DatasetSelectionConfig",
    "DirectionEvaluationResult",
    "ElasticNetGridSearchConfig",
    "ElasticNetIndependentBackend",
    "ElasticNetMultiOutputBackend",
    "EvaluationDirection",
    "EvaluationProtocol",
    "EvaluationProtocolDefinition",
    "FittedDirectionModel",
    "FittedCalibratedPixelModel",
    "FittedCalibratedPixelModels",
    "FittedIndependentRegressionModel",
    "FittedIndependentRegressionModels",
    "FittedMultiOutputRegressionModel",
    "LinearSVMBackend",
    "LinearSVMGridSearchConfig",
    "LoadedModelRun",
    "LogisticRegressionBackend",
    "MatrixRunSpec",
    "ModelPrediction",
    "ModelProtocolWorkflowResult",
    "ModelSpec",
    "MultiTargetSelectKBest",
    "PLSRegressionGridSearchConfig",
    "PLSRegressionMultiOutputBackend",
    "PixelTargetDataset",
    "PlattCalibration",
    "PlattCalibrationConfig",
    "PredictionMetrics",
    "ProtocolEvaluationResult",
    "ProtocolLeakageAudit",
    "RandomImageryExperimentConfigLike",
    "RandomImageryModelConfig",
    "RandomImageryModelBackend",
    "RandomForestIndependentBackend",
    "RandomForestMultiOutputBackend",
    "RandomForestRegressionGridSearchConfig",
    "RegressionCandidateScore",
    "RegressionExperimentConfig",
    "RegressionFeatureScreeningConfig",
    "RegressionFeatureScreeningResult",
    "RegressionGridSearchConfig",
    "RidgeClassifierBackend",
    "RidgeClassifierGridSearchConfig",
    "RidgeRegressionGridSearchConfig",
    "RidgeRegressionIndependentBackend",
    "RidgeRegressionMultiOutputBackend",
    "ScoreDiagnostics",
    "SubjectBootstrapInterval",
    "SubjectSplitConfig",
    "build_evaluation_protocol",
    "build_aligned_feature_partition",
    "build_model_backend",
    "build_model_run_hash",
    "build_random_imagery_targets",
    "evaluate_prediction_matrix",
    "compare_runs",
    "build_classical_matrix_plan",
    "build_matrix_plan_payload",
    "execute_classical_matrix_sweep",
    "execute_model_protocol",
    "fit_calibrated_pixel_classifiers",
    "fit_independent_regression_models",
    "fit_multioutput_regression_model",
    "get_model_spec",
    "load_calibrated_classifier_config",
    "load_model_config",
    "load_model_run",
    "load_regression_config",
    "parse_dotted_overrides",
    "model_run_dir",
    "replay_model_predictions",
    "run_model_evaluation_protocol",
    "screen_classifier_feature_families",
    "screen_regression_feature_families",
    "summarize_model_runs",
    "write_model_protocol_runs",
]
