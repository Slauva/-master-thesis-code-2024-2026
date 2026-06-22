from typing import Any

import numpy as np
import pandas as pd

from experiments.bnci2014_001 import (
    build_epoch_dataset,
    create_leave_one_subject_splits,
    fit_predict_torch_fft_pilot,
    fit_tensor_standardizer,
    load_bnci_config,
    select_validation_indices,
)


def _toy_bnci_dataset() -> Any:
    rows = []
    labels = []
    for subject in (1, 2, 3):
        for label in ("left_hand", "right_hand", "feet", "tongue"):
            rows.append({"subject": subject, "session": "0train", "run": "0"})
            labels.append(label)
    X = np.zeros((len(rows), 4, 1001), dtype=np.float32)
    return build_epoch_dataset(X, labels, pd.DataFrame(rows))


def test_torch_pilot_validation_subject_is_selected_inside_train_fold() -> None:
    dataset = _toy_bnci_dataset()
    config = load_bnci_config(overrides={"dataset": {"subjects": [1, 2, 3]}})
    split = create_leave_one_subject_splits(dataset)[0]

    train_fit, validation, validation_subject = select_validation_indices(
        dataset,
        split,
        split_config=config.split,
    )

    assert split.test_subjects == (1,)
    assert validation_subject == 2
    assert set(dataset.subjects[validation].tolist()) == {2}
    assert set(dataset.subjects[train_fit].tolist()) == {3}
    assert set(dataset.y[validation].tolist()) == {0, 1, 2, 3}
    assert set(dataset.y[train_fit].tolist()) == {0, 1, 2, 3}


def test_torch_pilot_standardizer_uses_train_tensor_statistics() -> None:
    train = np.asarray([[[[1.0, 3.0]]], [[[5.0, 7.0]]]], dtype=np.float32)

    mean, std = fit_tensor_standardizer(train)

    np.testing.assert_allclose(mean, np.asarray([[[[3.0, 5.0]]]], dtype=np.float32))
    np.testing.assert_allclose(std, np.asarray([[[[2.0, 2.0]]]], dtype=np.float32))


def test_torch_pilot_materializes_test_tensors_after_fit(
    monkeypatch: Any,
) -> None:
    dataset = _toy_bnci_dataset()
    config = load_bnci_config(
        overrides={
            "dataset": {"subjects": [1, 2, 3]},
            "torch_pilot": {"max_epochs": 2, "patience": 1},
        }
    )
    split = create_leave_one_subject_splits(dataset)[0]
    calls: list[str] = []

    def fake_materialize(_: Any, indices: np.ndarray, **__: object) -> np.ndarray:
        subjects = set(dataset.subjects[indices].tolist())
        if subjects == {3}:
            calls.append("train_fit_tensor")
        elif subjects == {2}:
            calls.append("validation_tensor")
        elif subjects == {1}:
            calls.append("test_tensor")
        else:
            calls.append("unexpected_tensor")
        return np.ones((indices.shape[0], 1, 4, 5), dtype=np.float32)

    def fake_train(*_: object, **__: object) -> tuple[list[dict[str, float]], int]:
        calls.append("fit")
        return ([{"epoch": 1.0, "validation_balanced_accuracy": 0.25}], 1)

    def fake_predict(*args: object, **__: object) -> np.ndarray:
        X = args[1]
        assert isinstance(X, np.ndarray)
        calls.append("predict")
        return np.full((X.shape[0], 4), 0.25, dtype=np.float64)

    monkeypatch.setattr("experiments.bnci2014_001.torch_pilot.materialize_fft_tensors", fake_materialize)
    monkeypatch.setattr("experiments.bnci2014_001.torch_pilot.train_torch_model", fake_train)
    monkeypatch.setattr("experiments.bnci2014_001.torch_pilot.predict_torch_probabilities", fake_predict)

    fit_predict_torch_fft_pilot(
        dataset,
        split,
        pilot_config=config.torch_pilot,
        split_config=config.split,
        source_sfreq=config.dataset.source_sfreq,
    )

    assert calls == [
        "train_fit_tensor",
        "validation_tensor",
        "fit",
        "test_tensor",
        "predict",
    ]
