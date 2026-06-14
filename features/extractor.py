from features.classical import (
    extract_spatial_features,
    extract_spectral_features,
    extract_time_features,
    prepare_feature_windows,
)
from features.config import FeatureExtractionConfig
from features.local_patterns import extract_local_pattern_features
from features.schemas import FeatureBlock, FeatureSet
from utils.datasets.schemas import LoadedSample


def extract_feature_set(
    loaded: LoadedSample,
    *,
    config: FeatureExtractionConfig,
) -> FeatureSet:
    """Extract configured feature groups from one canonical dataset block."""
    windows = prepare_feature_windows(
        loaded.eeg,
        source_sfreq=loaded.sfreq,
        config=config,
    )
    blocks: list[FeatureBlock] = []
    for group in config.feature_groups:
        if group == "time":
            blocks.append(extract_time_features(windows, dtype=config.dtype))
        elif group == "spectral":
            blocks.append(extract_spectral_features(windows, config=config))
        elif group == "spatial":
            blocks.extend(extract_spatial_features(windows, dtype=config.dtype))
        elif group == "local_patterns":
            blocks.extend(extract_local_pattern_features(windows, config=config))
        else:
            raise ValueError(f"Unsupported feature group: {group!r}")

    return FeatureSet(
        sample=loaded.sample,
        blocks=tuple(blocks),
        window_bounds_seconds=windows.bounds_seconds,
        eeg_channels=loaded.eeg_channels,
        analysis_sfreq=config.analysis_sfreq,
    )
