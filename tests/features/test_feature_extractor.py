from pathlib import Path

import numpy as np

from features import extract_feature_set, load_feature_config
from utils.datasets import GeometricSample, LoadedSample


def test_extract_feature_set_respects_group_order_and_source_metadata() -> None:
    sfreq = 125.0
    times = np.arange(251, dtype=np.float64) / sfreq
    loaded = LoadedSample(
        sample=GeometricSample(
            subject_id=2,
            trial_number=3,
            Exec_Block_Index=4,
            eeg_path=Path("exec_EEG_4.fif"),
            eog_path=Path("exec_EOG_4.fif"),
            img=[[0, 1], [1, 0]],
            pattern_id=5,
        ),
        eeg=np.stack(
            (
                np.sin(2 * np.pi * 10 * times),
                np.sin(2 * np.pi * 20 * times),
            )
        ).astype(np.float32),
        eog=np.zeros((1, times.size), dtype=np.float32),
        sfreq=sfreq,
        eeg_channels=("Fz", "Cz"),
        eog_channels=("EOG_x",),
    )
    config = load_feature_config(
        overrides={
            "crop_start_seconds": 0.0,
            "crop_end_seconds": 2.0,
            "window_seconds": 1.0,
            "window_stride_seconds": 1.0,
            "feature_groups": ["local_patterns", "time"],
        }
    )

    feature_set = extract_feature_set(loaded, config=config)

    assert feature_set.sample is loaded.sample
    assert feature_set.eeg_channels == loaded.eeg_channels
    assert feature_set.analysis_sfreq == 125.0
    assert tuple(block.name for block in feature_set.blocks) == ("lndp", "lgp", "lbp", "time")
    assert feature_set.blocks[0].values.shape == (2, 2, 256)
    assert feature_set.blocks[-1].values.shape == (2, 2, 13)
    np.testing.assert_array_equal(feature_set.window_bounds_seconds, [[0.0, 1.0], [1.0, 2.0]])
