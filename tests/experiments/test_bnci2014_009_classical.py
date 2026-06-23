import numpy as np
import pandas as pd
import pytest

from experiments.bnci2014_009 import (
    audit_split,
    build_epoch_dataset,
    build_erp_feature_matrix,
    create_leave_one_subject_splits,
    load_bnci009_config,
)
from experiments.bnci2014_009.baselines import fit_predict_classical_variant
from experiments.bnci2014_009.metrics import evaluate_binary_p300_predictions
from experiments.bnci2014_009.workflow import (
    build_classical_config_hash,
    run_classical_benchmark,
    validate_classical_manifest,
    write_classical_benchmark,
)


def _toy_dataset() -> tuple:
    rng = np.random.default_rng(42)
    rows = []
    labels = []
    epochs = []
    for subject in (1, 2, 3):
        for trial in range(18):
            label = "Target" if trial % 6 == 0 else "NonTarget"
            base = 1.0 if label == "Target" else -0.2
            signal = rng.normal(loc=base, scale=0.1, size=(16, 32))
            rows.append({"subject": subject, "session": "0", "run": "0"})
            labels.append(label)
            epochs.append(signal)
    return build_epoch_dataset(np.asarray(epochs), labels, pd.DataFrame(rows), dtype="float32")


def test_binary_p300_metrics_use_target_as_positive_score() -> None:
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.asarray([0, 1, 1, 1], dtype=np.int64)
    target_score = np.asarray([0.9, 0.8, 0.4, 0.1], dtype=np.float64)

    metrics = evaluate_binary_p300_predictions(
        y_true,
        y_pred,
        target_score=target_score,
        class_names=("Target", "NonTarget"),
    )

    assert metrics.target_recall == pytest.approx(0.5)
    assert metrics.non_target_recall == pytest.approx(1.0)
    assert metrics.balanced_accuracy == pytest.approx(0.75)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)


def test_fit_predict_classical_variants_preserve_split_boundary() -> None:
    dataset = _toy_dataset()
    erp = build_erp_feature_matrix(
        dataset,
        source_sfreq=32.0,
        waveform_stride=4,
        channel_names=tuple(f"C{index}" for index in range(16)),
    )
    config = load_bnci009_config(
        overrides={
            "dataset": {"source_sfreq": 32.0},
            "classical": {"variants": ["dummy-prior", "erp-logreg", "xdawn-tangent-logreg"]},
        }
    )
    split = create_leave_one_subject_splits(dataset)[0]
    audit = audit_split(dataset, split)

    assert not audit.has_forbidden_leakage
    for model_id in config.classical.variants:
        prediction = fit_predict_classical_variant(
            model_id,
            dataset,
            erp,
            split,
            classical_config=config.classical,
            split_config=config.split,
        )
        assert prediction.test_indices.tolist() == split.test_indices.tolist()
        assert prediction.y_true.shape == prediction.y_pred.shape == (split.test_indices.size,)
        assert prediction.target_score is not None
        assert prediction.target_score.shape == (split.test_indices.size,)


def test_run_classical_benchmark_subset_and_manifest(tmp_path) -> None:
    dataset = _toy_dataset()
    config = load_bnci009_config(
        overrides={
            "dataset": {"source_sfreq": 32.0},
            "classical": {"variants": ["dummy-prior", "erp-lda"]},
            "artifacts": {"root": str(tmp_path)},
        }
    )

    result = run_classical_benchmark(config, dataset=dataset)
    run_dir = write_classical_benchmark(config, result)
    validate_classical_manifest(run_dir)

    assert result.config_hash == build_classical_config_hash(config)
    assert [variant.model_id for variant in result.variants] == ["dummy-prior", "erp-lda"]
    assert all(variant.summary["n_folds"] == 3 for variant in result.variants)
    assert (run_dir / "evaluation.json").is_file()
    assert (run_dir / "predictions.json").is_file()
