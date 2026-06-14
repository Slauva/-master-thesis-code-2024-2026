from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import DTypeLike

from preprocessors.config import (
    FFTConfig,
    MorletConfig,
    PreprocessingConfig,
    PreprocessingMethod,
    STFTConfig,
    SuperletConfig,
    build_frequency_grid,
    load_preprocessing_config,
)
from preprocessors.schemas import SpectralTransformResult
from utils.datasets.base import SampleKey, SourceMap
from utils.datasets.numpy_dataset import NumpyDataset
from utils.datasets.schemas import LoadedSample, Sample, SpectralSample


class PreprocessedDataset(ABC):
    METHOD: ClassVar[PreprocessingMethod]
    CONFIG_TYPE: ClassVar[type[PreprocessingConfig]]

    def __init__(
        self,
        dataset_dir: Path,
        *,
        dataset_step_type: Literal["exec", "patt"] = "exec",
        dataset_pattern_type: Literal["geometric", "random"] | None = None,
        config: PreprocessingConfig | None = None,
        config_dir: Path | None = None,
        config_overrides: dict[str, object] | None = None,
        source_dtype: DTypeLike = np.float32,
        source_cache_policy: Literal["none", "memory", "disk", "both"] | None = "disk",
        source_cache_dir: Path | None = None,
        source_memory_cache_bytes: int = 1 << 30,
        exclude_samples: dict[str, list[str]] | None = None,
    ):
        if config is not None and (config_dir is not None or config_overrides is not None):
            raise ValueError("Pass either `config` or config loading options, not both")

        resolved_config = config or load_preprocessing_config(
            self.METHOD,
            config_dir=config_dir,
            overrides=config_overrides,
        )
        if not isinstance(resolved_config, self.CONFIG_TYPE):
            raise TypeError(f"{type(self).__name__} requires {self.CONFIG_TYPE.__name__}")

        self.config = resolved_config
        self.source_dataset = NumpyDataset(
            dataset_dir=dataset_dir,
            dataset_step_type=dataset_step_type,
            dataset_pattern_type=dataset_pattern_type,
            dtype=source_dtype,
            cache_policy=source_cache_policy,
            cache_dir=source_cache_dir,
            memory_cache_bytes=source_memory_cache_bytes,
            exclude_samples=exclude_samples,
        )

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, key: int | SampleKey) -> SpectralSample:
        loaded = self.source_dataset[key]
        self._validate_loaded_sample(loaded)
        transformed = self._transform(loaded)
        return self._build_spectral_sample(loaded, transformed)

    def __iter__(self) -> Iterator[SpectralSample]:
        for index in range(len(self)):
            yield self[index]

    @property
    def samples(self) -> tuple[Sample, ...]:
        return self.source_dataset.samples

    @property
    def source_map(self) -> SourceMap:
        return self.source_dataset.source_map

    @abstractmethod
    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        raise NotImplementedError

    def _build_spectral_sample(
        self,
        loaded: LoadedSample,
        transformed: SpectralTransformResult,
    ) -> SpectralSample:
        if transformed.eeg_power.shape[0] != len(loaded.eeg_channels):
            raise ValueError("The transformed EEG channel axis does not match the source EEG channels")
        if transformed.scaling != self.config.scaling:
            raise ValueError("Transform scaling does not match the preprocessing configuration")
        if not np.isclose(transformed.analysis_sfreq, self.config.analysis_sfreq):
            raise ValueError("Transform sampling frequency does not match the preprocessing configuration")
        expected_frequencies = build_frequency_grid(self.config)
        if transformed.frequencies.shape != expected_frequencies.shape or not np.allclose(
            transformed.frequencies,
            expected_frequencies,
        ):
            raise ValueError("Transform frequencies do not match the preprocessing configuration")

        output_dtype = np.dtype(self.config.dtype)
        return SpectralSample(
            sample=loaded.sample,
            eeg_power=transformed.eeg_power.astype(output_dtype, copy=False),
            eog=loaded.eog,
            frequencies=transformed.frequencies.astype(output_dtype, copy=False),
            times=None if transformed.times is None else transformed.times.astype(output_dtype, copy=False),
            eeg_channels=loaded.eeg_channels,
            eog_channels=loaded.eog_channels,
            source_sfreq=loaded.sfreq,
            analysis_sfreq=transformed.analysis_sfreq,
            method=self.METHOD,
            scaling=transformed.scaling,
        )

    @staticmethod
    def _validate_loaded_sample(loaded: LoadedSample) -> None:
        if loaded.eeg.ndim != 2:
            raise ValueError("Source EEG must have shape (channel, time)")
        if not np.isfinite(loaded.eeg).all():
            raise ValueError(
                "Source EEG contains non-finite values for "
                f"subject={loaded.sample.subject_id}, trial={loaded.sample.trial_number}, "
                f"block={loaded.sample.block_index}"
            )


class FFTDataset(PreprocessedDataset):
    METHOD = "fft"
    CONFIG_TYPE = FFTConfig

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        raise NotImplementedError("FFT transform is implemented at spectral preprocessing checkpoint 4")


class MorletDataset(PreprocessedDataset):
    METHOD = "morlet"
    CONFIG_TYPE = MorletConfig

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        raise NotImplementedError("Morlet transform is implemented at spectral preprocessing checkpoint 5")


class SuperletDataset(PreprocessedDataset):
    METHOD = "superlet"
    CONFIG_TYPE = SuperletConfig

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        raise NotImplementedError("Superlet transform is implemented at spectral preprocessing checkpoint 6")


class STFTDataset(PreprocessedDataset):
    METHOD = "stft"
    CONFIG_TYPE = STFTConfig

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        raise NotImplementedError("STFT transform is implemented at spectral preprocessing checkpoint 7")
