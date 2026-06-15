import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC

from experiments.random_imagery import (
    LinearSVMBackend,
    RidgeClassifierBackend,
    load_calibrated_classifier_config,
    run_model_evaluation_protocol,
)
from tests.experiments.test_logistic_regression_runner import _protocol_inputs


def _classifier_config(model_id: str) -> object:
    regularization_grid = (
        {"c_values": [0.1, 1.0]}
        if model_id == "linear-svm-independent"
        else {"alpha_values": [0.1, 1.0]}
    )
    return load_calibrated_classifier_config(
        model_id,
        overrides={
            "split": {"test_size": 0.25},
            "cross_validation": {"n_splits": 3},
            "feature_screening": {
                "select_k": 2,
                "regularization": 1.0,
                "candidates": [["time"], ["spectral"]],
            },
            "grid_search": {
                "select_k": [2],
                **regularization_grid,
                "class_weights": [None, "balanced"],
                "n_jobs": 1,
            },
            "bootstrap_iterations": 50,
        },
    )


@pytest.mark.parametrize(
    ("model_id", "backend", "estimator_type"),
    [
        ("linear-svm-independent", LinearSVMBackend(), LinearSVC),
        (
            "ridge-classifier-independent",
            RidgeClassifierBackend(),
            RidgeClassifier,
        ),
    ],
)
def test_calibrated_classifier_backends_use_grouped_oof_platt_scores(
    model_id: str,
    backend: object,
    estimator_type: type[object],
) -> None:
    dataset, targets = _protocol_inputs()
    config = _classifier_config(model_id)
    result = run_model_evaluation_protocol(
        "cross-subject",
        config=config,
        backend=backend,
        dataset=dataset,
        targets=targets,
    )

    direction = result.directions[0]
    prediction = direction.prediction
    fitted = direction.fitted_model.payload
    assert result.model_id == model_id
    assert prediction.diagnostics.score_semantics == "calibrated_probability"
    assert prediction.scores.dtype == np.dtype(np.float64)
    assert np.isfinite(prediction.scores).all()
    assert np.all((prediction.scores >= 0.0) & (prediction.scores <= 1.0))
    np.testing.assert_array_equal(
        prediction.predictions,
        (prediction.scores >= config.prediction_threshold).astype(np.int8),
    )

    assert len(fitted.models) == targets.y.shape[1]
    for pixel_model in fitted.models:
        assert isinstance(pixel_model.pipeline.named_steps["model"], estimator_type)
        assert pixel_model.pipeline.memory is None
        assert len(pixel_model.candidate_scores) == 4
        calibration = pixel_model.calibration
        assert calibration.oof_decision_scores.shape == (
            direction.direction.train_indices.size,
        )
        assert set(calibration.oof_fold_indices.tolist()) == {0, 1, 2}
        for fold in fitted.cross_validation.for_pixel(pixel_model.pixel_index):
            np.testing.assert_array_equal(
                calibration.oof_fold_indices[fold.validation_indices],
                np.full(fold.validation_indices.size, fold.fold_index),
            )
            assert not set(fold.train_subjects) & set(fold.validation_subjects)


@pytest.mark.parametrize(
    ("model_id", "backend"),
    [
        ("linear-svm-independent", LinearSVMBackend()),
        ("ridge-classifier-independent", RidgeClassifierBackend()),
    ],
)
def test_calibrated_classifier_protocol_is_deterministic(
    model_id: str,
    backend: object,
) -> None:
    config = _classifier_config(model_id)

    def run() -> object:
        dataset, targets = _protocol_inputs()
        return run_model_evaluation_protocol(
            "cross-subject",
            config=config,
            backend=backend,
            dataset=dataset,
            targets=targets,
        )

    first = run()
    second = run()
    first_direction = first.directions[0]
    second_direction = second.directions[0]
    np.testing.assert_allclose(
        first_direction.prediction.scores,
        second_direction.prediction.scores,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        first_direction.prediction.predictions,
        second_direction.prediction.predictions,
    )
    assert (
        first_direction.fitted_model.selection.selected_block_names
        == second_direction.fitted_model.selection.selected_block_names
    )
    assert tuple(
        model.best_hyperparameters
        for model in first_direction.fitted_model.payload.models
    ) == tuple(
        model.best_hyperparameters
        for model in second_direction.fitted_model.payload.models
    )


def test_classifier_runner_delays_test_features_until_all_calibrators_are_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.random_imagery import classifier_backend as classifier_module
    from experiments.random_imagery import runner as runner_module

    dataset, targets = _protocol_inputs()
    config = _classifier_config("linear-svm-independent")
    events: list[str] = []
    original_calibration = classifier_module._fit_oof_platt_calibration
    original_test_features = runner_module.build_aligned_feature_partition

    def record_calibration(*args: object, **kwargs: object) -> object:
        events.append(f"calibrate:{kwargs['pixel_index']}")
        return original_calibration(*args, **kwargs)

    def record_test_features(*args: object, **kwargs: object) -> object:
        events.append("test-features")
        return original_test_features(*args, **kwargs)

    monkeypatch.setattr(
        classifier_module,
        "_fit_oof_platt_calibration",
        record_calibration,
    )
    monkeypatch.setattr(
        runner_module,
        "build_aligned_feature_partition",
        record_test_features,
    )
    run_model_evaluation_protocol(
        "cross-subject",
        config=config,
        backend=LinearSVMBackend(),
        dataset=dataset,
        targets=targets,
    )

    assert events == ["calibrate:0", "calibrate:1", "test-features"]


def test_classifier_config_rejects_backend_grid_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match requested model"):
        load_calibrated_classifier_config(
            "linear-svm-independent",
            config_path="confs/experiments/ridge_classifier.yaml",
        )

    dataset, targets = _protocol_inputs()
    config = _classifier_config("ridge-classifier-independent")
    with pytest.raises(ValueError, match="does not match config model"):
        run_model_evaluation_protocol(
            "cross-subject",
            config=config,
            backend=LinearSVMBackend(),
            dataset=dataset,
            targets=targets,
        )
