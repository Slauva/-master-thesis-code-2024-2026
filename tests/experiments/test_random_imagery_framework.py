import numpy as np
import pytest

from experiments.logistic_regression import run_evaluation_protocol
from experiments.logistic_regression.config import (
    CrossValidationConfig as LogisticCrossValidationConfig,
)
from experiments.logistic_regression.config import (
    DatasetSelectionConfig as LogisticDatasetSelectionConfig,
)
from experiments.random_imagery import (
    MODEL_REGISTRY,
    PLANNED_MODEL_IDS,
    CrossValidationConfig,
    DatasetSelectionConfig,
    LogisticRegressionBackend,
    ModelPrediction,
    ScoreDiagnostics,
    run_model_evaluation_protocol,
)
from tests.experiments.test_logistic_regression_runner import (
    _protocol_inputs,
    _runner_config,
)


def test_model_registry_describes_reference_and_nine_planned_variants() -> None:
    assert len(MODEL_REGISTRY) == 10
    assert len(PLANNED_MODEL_IDS) == 9
    assert MODEL_REGISTRY["logistic-regression-independent"].reference is True
    assert MODEL_REGISTRY["pls-regression-multioutput"].exploratory is True
    assert {
        spec.topology for spec in MODEL_REGISTRY.values()
    } == {"independent", "multioutput"}
    assert {
        spec.task for spec in MODEL_REGISTRY.values()
    } == {"classifier", "regressor"}
    assert all(
        MODEL_REGISTRY[model_id].reference is False
        for model_id in PLANNED_MODEL_IDS
    )


def test_logistic_config_uses_shared_common_contracts() -> None:
    assert LogisticDatasetSelectionConfig is DatasetSelectionConfig
    assert LogisticCrossValidationConfig is CrossValidationConfig


def test_model_prediction_requires_bounded_scores_and_exact_thresholding() -> None:
    indices = np.asarray([0, 1], dtype=np.int64)
    scores = np.asarray([[0.25, 0.5], [0.75, 0.49]], dtype=np.float64)
    predictions = (scores >= 0.5).astype(np.int8)
    result = ModelPrediction(
        test_target_indices=indices,
        test_sample_keys=((1, 1, 1), (1, 1, 2)),
        scores=scores,
        predictions=predictions,
        threshold=0.5,
        diagnostics=ScoreDiagnostics(score_semantics="native_probability"),
    )
    np.testing.assert_array_equal(result.predictions, predictions)

    with pytest.raises(ValueError, match="thresholded"):
        ModelPrediction(
            test_target_indices=indices,
            test_sample_keys=((1, 1, 1), (1, 1, 2)),
            scores=scores,
            predictions=np.zeros_like(predictions),
            threshold=0.5,
            diagnostics=ScoreDiagnostics(score_semantics="native_probability"),
        )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ModelPrediction(
            test_target_indices=indices,
            test_sample_keys=((1, 1, 1), (1, 1, 2)),
            scores=np.asarray([[0.0, 1.1], [0.5, 0.5]], dtype=np.float64),
            predictions=predictions,
            threshold=0.5,
            diagnostics=ScoreDiagnostics(score_semantics="native_probability"),
        )


@pytest.mark.parametrize("protocol", ["cross-subject", "within-subject"])
def test_common_runner_reproduces_legacy_logistic_results(protocol: str) -> None:
    config = _runner_config()
    legacy_dataset, legacy_targets = _protocol_inputs()
    common_dataset, common_targets = _protocol_inputs()

    legacy = run_evaluation_protocol(
        protocol,  # type: ignore[arg-type]
        config=config,
        dataset=legacy_dataset,
        targets=legacy_targets,
    )
    common = run_model_evaluation_protocol(
        protocol,  # type: ignore[arg-type]
        config=config,
        backend=LogisticRegressionBackend(),
        dataset=common_dataset,
        targets=common_targets,
    )

    assert common.model_id == "logistic-regression-independent"
    assert len(common.directions) == len(legacy.directions)
    for common_direction, legacy_direction in zip(
        common.directions,
        legacy.directions,
        strict=True,
    ):
        assert (
            common_direction.fitted_model.selection.selected_block_names
            == legacy_direction.screening.selected_block_names
        )
        np.testing.assert_allclose(
            common_direction.prediction.scores,
            legacy_direction.grid_search.probabilities,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            common_direction.prediction.predictions,
            legacy_direction.grid_search.predictions,
        )
        np.testing.assert_allclose(
            common_direction.model_metrics.per_pixel_balanced_accuracy,
            legacy_direction.model_metrics.per_pixel_balanced_accuracy,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            common_direction.model_bootstrap.samples,
            legacy_direction.model_bootstrap.samples,
            rtol=0.0,
            atol=0.0,
        )
        assert (
            common_direction.model_metrics.mean_score_mse
            == legacy_direction.model_metrics.mean_brier_score
        )

    if protocol == "cross-subject":
        assert common.combined is None
        assert legacy.combined is None
    else:
        assert common.combined is not None
        assert legacy.combined is not None
        np.testing.assert_allclose(
            common.combined.prediction.scores,
            legacy.combined.probabilities,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            common.combined.prediction.predictions,
            legacy.combined.predictions,
        )
        np.testing.assert_allclose(
            common.combined.model_bootstrap.samples,
            legacy.combined.model_bootstrap.samples,
            rtol=0.0,
            atol=0.0,
        )


def test_common_runner_materializes_test_features_only_after_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.random_imagery import runner as runner_module

    dataset, targets = _protocol_inputs()
    config = _runner_config()
    backend = LogisticRegressionBackend()
    events: list[str] = []
    original_fit = backend.fit
    original_predict = backend.predict
    original_test_features = runner_module.build_aligned_feature_partition

    class RecordingBackend:
        spec = backend.spec

        def fit(self, *args: object, **kwargs: object) -> object:
            events.append("fit")
            return original_fit(*args, **kwargs)

        def predict(self, *args: object, **kwargs: object) -> object:
            events.append("predict")
            return original_predict(*args, **kwargs)

    def record_test_features(*args: object, **kwargs: object) -> object:
        events.append("test-features")
        return original_test_features(*args, **kwargs)

    monkeypatch.setattr(
        runner_module,
        "build_aligned_feature_partition",
        record_test_features,
    )
    run_model_evaluation_protocol(
        "cross-subject",
        config=config,
        backend=RecordingBackend(),
        dataset=dataset,
        targets=targets,
    )

    assert events == ["fit", "test-features", "predict"]
