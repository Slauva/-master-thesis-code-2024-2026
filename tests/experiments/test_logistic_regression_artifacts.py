import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.logistic_regression import (
    PixelTargetDataset,
    SubjectSplit,
    build_aligned_feature_partition,
    load_experiment_run,
    load_logistic_regression_config,
    reproduce_experiment_predictions,
    run_per_pixel_grid_search,
    write_experiment_run,
)
from features import FeatureBlock, FeatureSet
from utils.datasets import RandomSample


class SyntheticFeatureDataset:
    def __init__(self, feature_sets: dict[tuple[int, int, int], FeatureSet]) -> None:
        self.feature_sets = feature_sets

    def __len__(self) -> int:
        return len(self.feature_sets)

    def __getitem__(self, key: int | tuple[int, int, int]) -> FeatureSet:
        if isinstance(key, int):
            raise AssertionError("Artifact test must use canonical sample keys")
        return self.feature_sets[key]


def _sample(subject: int) -> RandomSample:
    return RandomSample(
        subject_id=subject,
        trial_number=1,
        Exec_Block_Index=1,
        eeg_path=Path(f"S_{subject}/patt_EEG_1.fif"),
        eog_path=Path(f"S_{subject}/patt_EOG_1.fif"),
        img=np.zeros((6, 6), dtype=np.int8).tolist(),
        seed=subject,
    )


def _targets() -> PixelTargetDataset:
    subject_ids = np.arange(1, 19, dtype=np.int64)
    y = np.column_stack((subject_ids % 2, (subject_ids // 2) % 2)).astype(np.int8)
    return PixelTargetDataset(
        y=y,
        pixel_names=("pixel_r0_c0", "pixel_r0_c1"),
        sample_keys=tuple((int(subject), 1, 1) for subject in subject_ids),
        subject_ids=subject_ids,
        trial_numbers=np.ones(subject_ids.size, dtype=np.int64),
        block_indices=np.ones(subject_ids.size, dtype=np.int64),
        seeds=subject_ids.copy(),
        image_fingerprints=tuple(f"fingerprint-{subject}" for subject in subject_ids),
    )


def _dataset(targets: PixelTargetDataset) -> SyntheticFeatureDataset:
    feature_sets: dict[tuple[int, int, int], FeatureSet] = {}
    for row_index, key in enumerate(targets.sample_keys):
        rng = np.random.default_rng(key[0])
        values = rng.normal(scale=0.05, size=8)
        values[:2] += 2.0 * targets.y[row_index]
        feature_sets[key] = FeatureSet(
            sample=_sample(key[0]),
            blocks=(
                FeatureBlock(
                    name="lbp",
                    layout="channel_histogram",
                    values=values.reshape(1, 1, -1).astype(np.float32),
                    feature_names=tuple(f"code_{index:03d}" for index in range(values.size)),
                ),
            ),
            window_bounds_seconds=np.asarray([[0.5, 15.5]], dtype=np.float64),
            eeg_channels=("Fz",),
            analysis_sfreq=125.0,
        )
    return SyntheticFeatureDataset(feature_sets)


def _split() -> SubjectSplit:
    return SubjectSplit(
        train_indices=np.arange(15, dtype=np.int64),
        test_indices=np.arange(15, 18, dtype=np.int64),
        train_subjects=tuple(range(1, 16)),
        test_subjects=(16, 17, 18),
        n_samples=18,
        random_state=42,
        test_size=1 / 6,
    )


def _build_run(tmp_path: Path) -> tuple[Path, object, object, object, object]:
    targets = _targets()
    dataset = _dataset(targets)
    split = _split()
    config = load_logistic_regression_config(
        overrides={
            "cross_validation": {"n_splits": 3},
            "grid_search": {
                "select_k": [2, 4],
                "c_values": [0.1, 1.0],
                "penalties": ["l1", "l2"],
                "class_weights": [None, "balanced"],
                "max_iter": 1000,
                "n_jobs": 1,
            },
            "artifacts": {
                "root": str(tmp_path / "runs"),
                "schema_version": 1,
            },
        }
    )
    result = run_per_pixel_grid_search(
        dataset,
        targets=targets,
        split=split,
        block_names=("lbp",),
        cross_validation_config=config.cross_validation,
        grid_search_config=config.grid_search,
        scoring=config.cross_validation.scoring,
        threshold=config.prediction_threshold,
        random_state=config.random_state,
    )
    run_dir = write_experiment_run(
        result,
        targets=targets,
        split=split,
        config=config,
        feature_config_hash="synthetic-feature-config",
    )
    return run_dir, result, targets, split, dataset


def test_round_trip_and_prediction_reproduction(tmp_path: Path) -> None:
    run_dir, result, targets, split, dataset = _build_run(tmp_path)

    with pytest.raises(PermissionError, match="trusted=True"):
        load_experiment_run(run_dir)
    loaded = load_experiment_run(run_dir, trusted=True)
    test_features = build_aligned_feature_partition(
        dataset,
        targets=targets,
        row_indices=split.test_indices,
        block_names=("lbp",),
    )
    probabilities, predictions = reproduce_experiment_predictions(
        loaded,
        test_features=test_features,
    )

    np.testing.assert_allclose(probabilities, result.probabilities, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(predictions, result.predictions)
    np.testing.assert_allclose(loaded.probabilities, result.probabilities, rtol=0.0, atol=0.0)
    assert len(loaded.pipelines) == 2
    assert loaded.manifest["file_count"] == 13
    assert "manifest.json" not in loaded.manifest["files"]
    assert loaded.environment["packages"]["scikit-learn"]


def test_duplicate_run_is_rejected(tmp_path: Path) -> None:
    run_dir, result, targets, split, _ = _build_run(tmp_path)
    config = load_logistic_regression_config(
        config_path=run_dir / "config.json"
    )

    with pytest.raises(FileExistsError, match="immutable"):
        write_experiment_run(
            result,
            targets=targets,
            split=split,
            config=config,
            feature_config_hash="synthetic-feature-config",
        )


@pytest.mark.parametrize(
    ("relative_path", "operation", "match"),
    [
        ("pipelines/pixel_00.joblib", "delete", "inventory mismatch"),
        ("arrays/probabilities.npy", "corrupt", "size mismatch|hash mismatch"),
    ],
)
def test_missing_or_corrupt_file_is_rejected(
    tmp_path: Path,
    relative_path: str,
    operation: str,
    match: str,
) -> None:
    run_dir, *_ = _build_run(tmp_path)
    path = run_dir / relative_path
    if operation == "delete":
        path.unlink()
    else:
        with path.open("ab") as file:
            file.write(b"corruption")

    with pytest.raises(ValueError, match=match):
        load_experiment_run(run_dir, trusted=True)


def test_pipeline_path_must_be_manifested_and_local(tmp_path: Path) -> None:
    run_dir, *_ = _build_run(tmp_path)
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["pipeline_files"][0] = "../../outside.joblib"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = results_path.read_bytes()
    manifest["files"]["results.json"] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe experiment pipeline"):
        load_experiment_run(run_dir, trusted=True)
