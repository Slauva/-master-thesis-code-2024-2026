from dataclasses import replace

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge

from experiments.random_imagery import (
    ElasticNetIndependentBackend,
    ElasticNetMultiOutputBackend,
    FittedIndependentRegressionModels,
    FittedMultiOutputRegressionModel,
    MultiTargetSelectKBest,
    PLSRegressionMultiOutputBackend,
    RandomForestIndependentBackend,
    RandomForestMultiOutputBackend,
    RidgeRegressionIndependentBackend,
    RidgeRegressionMultiOutputBackend,
    load_regression_config,
    run_model_evaluation_protocol,
)
from experiments.random_imagery.shared import build_aligned_feature_partition
from tests.experiments.test_logistic_regression_runner import _protocol_inputs

REGRESSION_CASES = (
    (
        "ridge-regression-independent",
        RidgeRegressionIndependentBackend(),
        Ridge,
        {"alpha_values": [1.0]},
    ),
    (
        "ridge-regression-multioutput",
        RidgeRegressionMultiOutputBackend(),
        Ridge,
        {"alpha_values": [1.0]},
    ),
    (
        "elastic-net-independent",
        ElasticNetIndependentBackend(),
        ElasticNet,
        {"alpha_values": [0.01], "l1_ratios": [0.5, 1.0]},
    ),
    (
        "elastic-net-multioutput",
        ElasticNetMultiOutputBackend(),
        ElasticNet,
        {"alpha_values": [0.01], "l1_ratios": [0.5, 1.0]},
    ),
    (
        "random-forest-independent",
        RandomForestIndependentBackend(),
        RandomForestRegressor,
        {
            "n_estimators": [10],
            "max_depth": [None],
            "min_samples_leaf": [1],
            "max_features": [1.0],
        },
    ),
    (
        "random-forest-multioutput",
        RandomForestMultiOutputBackend(),
        RandomForestRegressor,
        {
            "n_estimators": [10],
            "max_depth": [None],
            "min_samples_leaf": [1],
            "max_features": [1.0],
        },
    ),
    (
        "pls-regression-multioutput",
        PLSRegressionMultiOutputBackend(),
        PLSRegression,
        {"n_components": [1]},
    ),
)


def _regression_config(
    model_id: str,
    grid_overrides: dict[str, object],
) -> object:
    return load_regression_config(
        model_id,
        overrides={
            "split": {"test_size": 0.25},
            "cross_validation": {"n_splits": 3},
            "feature_screening": {
                "select_k": 2,
                "candidates": [["time"], ["spectral"]],
            },
            "grid_search": {
                "select_k": [2],
                **grid_overrides,
                "n_jobs": 1,
            },
            "bootstrap_iterations": 20,
        },
    )


@pytest.mark.parametrize(
    ("model_id", "backend", "estimator_type", "grid_overrides"),
    REGRESSION_CASES,
)
def test_regression_backends_produce_bounded_scores_and_expected_topology(
    model_id: str,
    backend: object,
    estimator_type: type[object],
    grid_overrides: dict[str, object],
) -> None:
    dataset, targets = _protocol_inputs()
    config = _regression_config(model_id, grid_overrides)
    result = run_model_evaluation_protocol(
        "cross-subject",
        config=config,
        backend=backend,
        dataset=dataset,
        targets=targets,
    )

    direction = result.directions[0]
    prediction = direction.prediction
    payload = direction.fitted_model.payload
    assert prediction.scores.shape == (18, 2)
    assert prediction.scores.dtype == np.dtype(np.float64)
    assert prediction.diagnostics.score_semantics == "clipped_regression"
    assert np.isfinite(prediction.scores).all()
    assert np.all((prediction.scores >= 0.0) & (prediction.scores <= 1.0))
    np.testing.assert_array_equal(
        prediction.predictions,
        (prediction.scores >= config.prediction_threshold).astype(np.int8),
    )
    assert 0.0 <= prediction.diagnostics.clipped_below_zero_fraction <= 1.0
    assert 0.0 <= prediction.diagnostics.clipped_above_one_fraction <= 1.0

    if model_id.endswith("-independent"):
        assert isinstance(payload, FittedIndependentRegressionModels)
        assert len(payload.models) == 2
        estimators = tuple(model.pipeline.named_steps["model"] for model in payload.models)
    else:
        assert isinstance(payload, FittedMultiOutputRegressionModel)
        estimators = (payload.pipeline.named_steps["model"],)
        assert isinstance(payload.pipeline.named_steps["select"], MultiTargetSelectKBest)
        for fold in payload.cross_validation.folds:
            assert not set(fold.train_subjects) & set(fold.validation_subjects)
            for target_index in range(targets.y.shape[1]):
                assert np.unique(
                    targets.y[
                        direction.direction.train_indices[fold.train_indices],
                        target_index,
                    ]
                ).size == 2
                assert np.unique(
                    targets.y[
                        direction.direction.train_indices[fold.validation_indices],
                        target_index,
                    ]
                ).size == 2
    assert all(isinstance(estimator, estimator_type) for estimator in estimators)
    assert all(
        not isinstance(estimator, RandomForestRegressor) or estimator.n_jobs == 1
        for estimator in estimators
    )


@pytest.mark.parametrize(
    ("model_id", "backend", "_estimator_type", "grid_overrides"),
    REGRESSION_CASES,
)
def test_regression_backends_are_deterministic(
    model_id: str,
    backend: object,
    _estimator_type: type[object],
    grid_overrides: dict[str, object],
) -> None:
    config = _regression_config(model_id, grid_overrides)

    def run() -> object:
        dataset, targets = _protocol_inputs()
        return run_model_evaluation_protocol(
            "cross-subject",
            config=config,
            backend=backend,
            dataset=dataset,
            targets=targets,
        )

    first = run().directions[0]
    second = run().directions[0]
    np.testing.assert_allclose(
        first.prediction.scores,
        second.prediction.scores,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        first.prediction.predictions,
        second.prediction.predictions,
    )
    assert (
        first.fitted_model.selection.selected_block_names
        == second.fitted_model.selection.selected_block_names
    )


@pytest.mark.parametrize(
    ("model_id", "backend", "grid_overrides"),
    (
        (
            "ridge-regression-independent",
            RidgeRegressionIndependentBackend(),
            {"alpha_values": [1.0]},
        ),
        (
            "ridge-regression-multioutput",
            RidgeRegressionMultiOutputBackend(),
            {"alpha_values": [1.0]},
        ),
    ),
)
def test_regression_topologies_support_exactly_36_targets(
    model_id: str,
    backend: object,
    grid_overrides: dict[str, object],
) -> None:
    dataset, base_targets = _protocol_inputs()
    y = np.tile(base_targets.y, (1, 18)).astype(np.int8)
    targets = replace(
        base_targets,
        y=y,
        pixel_names=tuple(f"pixel_{index:02d}" for index in range(36)),
    )
    config = _regression_config(model_id, grid_overrides)
    result = run_model_evaluation_protocol(
        "cross-subject",
        config=config,
        backend=backend,
        dataset=dataset,
        targets=targets,
    )

    assert result.directions[0].prediction.scores.shape == (18, 36)
    payload = result.directions[0].fitted_model.payload
    if model_id.endswith("-independent"):
        assert isinstance(payload, FittedIndependentRegressionModels)
        assert len(payload.models) == 36
    else:
        assert isinstance(payload, FittedMultiOutputRegressionModel)


def test_regression_clipping_diagnostics_match_raw_predictions() -> None:
    dataset, targets = _protocol_inputs()
    config = _regression_config(
        "ridge-regression-independent",
        {"alpha_values": [1.0]},
    )
    result = run_model_evaluation_protocol(
        "cross-subject",
        config=config,
        backend=RidgeRegressionIndependentBackend(),
        dataset=dataset,
        targets=targets,
    )
    direction = result.directions[0]
    payload = direction.fitted_model.payload
    assert isinstance(payload, FittedIndependentRegressionModels)
    test_features = build_aligned_feature_partition(
        dataset,
        targets=targets,
        row_indices=direction.direction.test_indices,
        block_names=direction.fitted_model.selected_block_names,
    )
    raw = np.column_stack(
        [
            model.pipeline.predict(test_features.X)
            for model in payload.models
        ]
    )

    np.testing.assert_allclose(
        direction.prediction.scores,
        np.clip(raw, 0.0, 1.0),
        rtol=0.0,
        atol=0.0,
    )
    assert direction.prediction.diagnostics.clipped_below_zero_fraction == float(
        np.mean(raw < 0.0)
    )
    assert direction.prediction.diagnostics.clipped_above_one_fraction == float(
        np.mean(raw > 1.0)
    )


def test_multi_target_selector_resolves_equal_ranks_by_feature_index() -> None:
    y = np.tile([[0, 1], [1, 0]], (6, 1)).astype(np.int8)
    signal = y[:, 0].astype(np.float64)
    X = np.column_stack((signal, signal, signal, 1.0 - signal))
    selector = MultiTargetSelectKBest(k=2).fit(X, y)

    np.testing.assert_array_equal(
        selector.get_support(indices=True),
        np.asarray([0, 1], dtype=np.int64),
    )
    assert selector.transform(X).shape == (12, 2)


def test_regression_grid_ties_use_mse_then_candidate_order() -> None:
    from experiments.random_imagery.regression_backend import (
        RegressionCandidateScreeningResult,
        _select_best_grid_candidate,
        _select_screening_candidate,
    )

    lower_mse = _select_best_grid_candidate(
        {
            "mean_test_balanced_accuracy": np.asarray([0.6, 0.6]),
            "mean_test_negative_clipped_mse": np.asarray([-0.3, -0.2]),
        }
    )
    exact_tie = _select_best_grid_candidate(
        {
            "mean_test_balanced_accuracy": np.asarray([0.6, 0.6]),
            "mean_test_negative_clipped_mse": np.asarray([-0.2, -0.2]),
        }
    )

    assert lower_mse == 1
    assert exact_tie == 0

    def screening_candidate(
        name: str,
        *,
        balanced_accuracy: float,
        clipped_mse: float,
    ) -> RegressionCandidateScreeningResult:
        return RegressionCandidateScreeningResult(
            block_names=(name,),
            fold_balanced_accuracy=np.asarray(
                [[balanced_accuracy]],
                dtype=np.float64,
            ),
            fold_clipped_mse=np.asarray([[clipped_mse]], dtype=np.float64),
            selected_feature_counts=np.asarray([[1]], dtype=np.int64),
            mean_balanced_accuracy=balanced_accuracy,
            mean_clipped_mse=clipped_mse,
        )

    lower_screening_mse = _select_screening_candidate(
        (
            screening_candidate("first", balanced_accuracy=0.6, clipped_mse=0.3),
            screening_candidate("second", balanced_accuracy=0.6, clipped_mse=0.2),
        )
    )
    exact_screening_tie = _select_screening_candidate(
        (
            screening_candidate("first", balanced_accuracy=0.6, clipped_mse=0.2),
            screening_candidate("second", balanced_accuracy=0.6, clipped_mse=0.2),
        )
    )
    assert lower_screening_mse == 1
    assert exact_screening_tie == 0


def test_regression_grid_scoring_uses_configured_prediction_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.random_imagery import regression_backend as regression_module

    dataset, targets = _protocol_inputs()
    config = load_regression_config(
        "ridge-regression-independent",
        overrides={
            "split": {"test_size": 0.25},
            "cross_validation": {"n_splits": 3},
            "feature_screening": {
                "select_k": 2,
                "candidates": [["time"]],
            },
            "grid_search": {
                "select_k": [2],
                "alpha_values": [1.0],
                "n_jobs": 1,
            },
            "prediction_threshold": 0.4,
            "bootstrap_iterations": 20,
        },
    )
    thresholds: list[float] = []
    original_scorer = regression_module._threshold_balanced_accuracy_scorer

    def record_threshold(*args: object, threshold: float, **kwargs: object) -> float:
        thresholds.append(threshold)
        return original_scorer(*args, threshold=threshold, **kwargs)

    monkeypatch.setattr(
        regression_module,
        "_threshold_balanced_accuracy_scorer",
        record_threshold,
    )
    result = run_model_evaluation_protocol(
        "cross-subject",
        config=config,
        backend=RidgeRegressionIndependentBackend(),
        dataset=dataset,
        targets=targets,
    )

    assert thresholds and set(thresholds) == {0.4}
    np.testing.assert_array_equal(
        result.directions[0].prediction.predictions,
        (result.directions[0].prediction.scores >= 0.4).astype(np.int8),
    )


@pytest.mark.parametrize(
    ("model_id", "backend", "grid_overrides"),
    (
        (
            "ridge-regression-independent",
            RidgeRegressionIndependentBackend(),
            {"alpha_values": [1.0]},
        ),
        (
            "pls-regression-multioutput",
            PLSRegressionMultiOutputBackend(),
            {"n_components": [1]},
        ),
    ),
)
def test_regression_runner_delays_test_features_until_fit_completes(
    model_id: str,
    backend: object,
    grid_overrides: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.random_imagery import runner as runner_module

    dataset, targets = _protocol_inputs()
    config = _regression_config(model_id, grid_overrides)
    events: list[str] = []
    original_fit = backend.fit
    original_test_features = runner_module.build_aligned_feature_partition

    class RecordingBackend:
        spec = backend.spec

        def fit(self, *args: object, **kwargs: object) -> object:
            result = original_fit(*args, **kwargs)
            events.append("fit-complete")
            return result

        def predict(self, *args: object, **kwargs: object) -> object:
            return backend.predict(*args, **kwargs)

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

    assert events == ["fit-complete", "test-features"]


def test_elastic_net_grid_requires_lasso_candidate() -> None:
    with pytest.raises(ValueError, match="l1_ratio=1.0"):
        load_regression_config(
            "elastic-net-independent",
            overrides={"grid_search": {"l1_ratios": [0.1, 0.5]}},
        )


def test_regression_screening_parameters_are_explicit_not_grid_ordered() -> None:
    from experiments.random_imagery.regression_backend import _screening_parameters

    config = load_regression_config(
        "ridge-regression-independent",
        overrides={
            "grid_search": {
                "alpha_values": [10.0, 1.0],
                "screening_alpha": 0.25,
            }
        },
    )

    assert _screening_parameters(config.grid_search) == {"alpha": 0.25}
