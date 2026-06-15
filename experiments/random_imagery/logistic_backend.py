from dataclasses import dataclass

from experiments.logistic_regression.config import LogisticRegressionExperimentConfig
from experiments.logistic_regression.modeling import (
    fit_pixel_models,
    predict_pixel_models,
)
from experiments.logistic_regression.schemas import (
    AlignedFeaturePartition,
    AlignedTrainingFeatures,
    FeatureScreeningResult,
    FittedPixelModels,
    PixelTargetDataset,
)
from experiments.logistic_regression.screening import (
    build_grouped_pixel_cross_validation,
    screen_feature_families,
)
from experiments.random_imagery.config import RandomImageryExperimentConfigLike
from experiments.random_imagery.contracts import (
    FittedDirectionModel,
    ModelPrediction,
    ScoreDiagnostics,
)
from experiments.random_imagery.registry import get_model_spec


@dataclass(frozen=True, slots=True)
class LogisticRegressionBackend:
    spec = get_model_spec("logistic-regression-independent")

    def fit(
        self,
        training_features: AlignedTrainingFeatures,
        *,
        targets: PixelTargetDataset,
        config: RandomImageryExperimentConfigLike,
    ) -> FittedDirectionModel[FittedPixelModels, FeatureScreeningResult]:
        logistic_config = _require_logistic_config(config)
        y_train = targets.y[training_features.target_row_indices]
        cross_validation = build_grouped_pixel_cross_validation(
            y_train,
            groups=training_features.subject_ids,
            config=logistic_config.cross_validation,
        )
        screening = screen_feature_families(
            training_features,
            y=y_train,
            cross_validation=cross_validation,
            config=logistic_config.feature_screening,
            random_state=logistic_config.random_state,
        )
        selected = _select_training_partition(
            training_features,
            block_names=screening.selected_block_names,
        )
        fitted_models = fit_pixel_models(
            selected,
            y_train=y_train,
            pixel_names=targets.pixel_names,
            cross_validation=cross_validation,
            config=logistic_config.grid_search,
            scoring=logistic_config.cross_validation.scoring,
            random_state=logistic_config.random_state,
        )
        return FittedDirectionModel(
            spec=self.spec,
            selected_block_names=screening.selected_block_names,
            feature_names=selected.feature_names,
            training_target_indices=selected.target_row_indices,
            training_sample_keys=selected.sample_keys,
            payload=fitted_models,
            selection=screening,
        )

    def predict(
        self,
        fitted: FittedDirectionModel[FittedPixelModels, FeatureScreeningResult],
        *,
        test_features: AlignedFeaturePartition,
        targets: PixelTargetDataset,
        config: RandomImageryExperimentConfigLike,
    ) -> ModelPrediction:
        logistic_config = _require_logistic_config(config)
        result = predict_pixel_models(
            fitted.payload,
            test_features=test_features,
            y_test=targets.y[test_features.target_row_indices],
            threshold=logistic_config.prediction_threshold,
        )
        return ModelPrediction(
            test_target_indices=result.test_target_indices,
            test_sample_keys=result.test_sample_keys,
            scores=result.probabilities,
            predictions=result.predictions,
            threshold=result.threshold,
            diagnostics=ScoreDiagnostics(
                score_semantics=self.spec.score_semantics,
            ),
        )


def _select_training_partition(
    features: AlignedTrainingFeatures,
    *,
    block_names: tuple[str, ...],
) -> AlignedFeaturePartition:
    matches = tuple(
        family for family in features.families if family.block_names == block_names
    )
    if len(matches) != 1:
        raise ValueError("Selected feature family is absent or ambiguous")
    family = matches[0]
    return AlignedFeaturePartition(
        block_names=family.block_names,
        X=family.X,
        feature_names=family.feature_names,
        target_row_indices=features.target_row_indices,
        sample_keys=features.sample_keys,
        subject_ids=features.subject_ids,
        window_bounds_seconds=features.window_bounds_seconds,
    )


def _require_logistic_config(
    config: RandomImageryExperimentConfigLike,
) -> LogisticRegressionExperimentConfig:
    if not isinstance(config, LogisticRegressionExperimentConfig):
        raise TypeError("Logistic Regression backend requires its experiment configuration")
    return config


__all__ = ["LogisticRegressionBackend"]
