import hashlib
import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

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
from preprocessors.fft import compute_fft_psd
from preprocessors.morlet import compute_morlet_power
from preprocessors.schemas import SpectralTransformResult
from preprocessors.stft import compute_stft_psd
from preprocessors.superlet import compute_superlet_power
from utils.datasets.base import SampleKey, SourceMap
from utils.datasets.numpy_dataset import NumpyDataset
from utils.datasets.schemas import LoadedSample, Sample, SpectralSample


class PreprocessedDataset(ABC):
    CACHE_SCHEMA_VERSION: ClassVar[int] = 1
    TRANSFORM_VERSION: ClassVar[int] = 1
    CACHE_MANIFEST_FILENAME: ClassVar[str] = "manifest.json"
    CACHE_EEG_POWER_FILENAME: ClassVar[str] = "eeg_power.npy"
    CACHE_FREQUENCIES_FILENAME: ClassVar[str] = "frequencies.npy"
    CACHE_TIMES_FILENAME: ClassVar[str] = "times.npy"
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
        cache_policy: Literal["none", "disk"] | None = "disk",
        cache_dir: Path | None = None,
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
        if cache_policy not in (None, "none", "disk"):
            raise ValueError(f"Unsupported spectral cache policy: {cache_policy!r}")

        self.config = resolved_config
        self.cache_policy = cache_policy
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
        self.source_dtype = self.source_dataset.dtype
        self.config_hash = self._build_config_hash()
        self.cache_dir = self._resolve_cache_dir(cache_dir) if self._uses_disk_cache else None

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, key: int | SampleKey) -> SpectralSample:
        loaded = self.source_dataset[key]
        self._validate_loaded_sample(loaded)
        if self._uses_disk_cache:
            cached = self._load_disk_cache(loaded)
            if cached is not None:
                return cached

        transformed = self._transform(loaded)
        spectral_sample = self._build_spectral_sample(loaded, transformed)
        if self._uses_disk_cache:
            self._write_disk_cache(spectral_sample)
        return spectral_sample

    def __iter__(self) -> Iterator[SpectralSample]:
        for index in range(len(self)):
            yield self[index]

    @property
    def _uses_disk_cache(self) -> bool:
        return self.cache_policy == "disk"

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
        expected_ndim = 2 if self.METHOD == "fft" else 3
        if transformed.eeg_power.ndim != expected_ndim:
            expected_shape = "(channel, frequency)" if expected_ndim == 2 else "(channel, frequency, time)"
            raise ValueError(f"{self.METHOD.upper()} power must have shape {expected_shape}")
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

    def get_cache_entry_path(self, key: int | SampleKey) -> Path:
        if self.cache_dir is None:
            raise ValueError("Spectral disk cache is disabled for this dataset")
        sample = self.source_dataset._get_sample(key)
        return (
            self.cache_dir
            / f"S_{sample.subject_id}"
            / f"Trial_{sample.trial_number}"
            / f"Block_{sample.block_index}"
        )

    def _resolve_cache_dir(self, cache_dir: Path | None) -> Path:
        if cache_dir is not None:
            return Path(cache_dir)

        dataset_dir = self.source_dataset.dataset_dir.resolve()
        for candidate in (dataset_dir, *dataset_dir.parents):
            if (candidate / "pyproject.toml").is_file():
                return (
                    candidate
                    / "artifacts"
                    / "preprocessed"
                    / self.source_dataset.dataset_dir.name
                    / self.source_dataset.dataset_step_type
                    / self.METHOD
                    / self.config_hash
                )
        raise ValueError("Could not determine default spectral cache directory; pass `cache_dir` explicitly")

    def _build_config_hash(self) -> str:
        canonical_payload = json.dumps(
            self._cache_identity(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical_payload).hexdigest()[:16]

    def _cache_identity(self) -> dict[str, Any]:
        return {
            "cache_schema_version": self.CACHE_SCHEMA_VERSION,
            "transform_version": self.TRANSFORM_VERSION,
            "transform_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "method": self.METHOD,
            "source_dtype": self.source_dtype.name,
            "config": self.config.model_dump(mode="json"),
        }

    def _load_disk_cache(self, loaded: LoadedSample) -> SpectralSample | None:
        entry_dir = self.get_cache_entry_path(
            (loaded.sample.subject_id, loaded.sample.trial_number, loaded.sample.block_index)
        )
        manifest_path = entry_dir / self.CACHE_MANIFEST_FILENAME
        eeg_power_path = entry_dir / self.CACHE_EEG_POWER_FILENAME
        frequencies_path = entry_dir / self.CACHE_FREQUENCIES_FILENAME
        if not (manifest_path.is_file() and eeg_power_path.is_file() and frequencies_path.is_file()):
            return None

        try:
            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            if not self._manifest_matches_loaded_sample(manifest, loaded):
                return None

            eeg_power = np.load(eeg_power_path, allow_pickle=False)
            frequencies = np.load(frequencies_path, allow_pickle=False)
            times_required = manifest["arrays"]["times"] is not None
            times_path = entry_dir / self.CACHE_TIMES_FILENAME
            if times_required and not times_path.is_file():
                return None
            times = np.load(times_path, allow_pickle=False) if times_required else None
            if not self._manifest_matches_arrays(
                manifest,
                eeg_power=eeg_power,
                frequencies=frequencies,
                times=times,
            ):
                return None

            transformed = SpectralTransformResult(
                eeg_power=eeg_power,
                frequencies=frequencies,
                times=times,
                analysis_sfreq=float(manifest["analysis_sfreq"]),
                scaling=manifest["scaling"],
            )
            return self._build_spectral_sample(loaded, transformed)
        except (EOFError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write_disk_cache(self, spectral_sample: SpectralSample) -> None:
        entry_dir = self.get_cache_entry_path(
            (
                spectral_sample.sample.subject_id,
                spectral_sample.sample.trial_number,
                spectral_sample.sample.block_index,
            )
        )
        entry_dir.mkdir(parents=True, exist_ok=True)

        self._atomic_save_array(entry_dir / self.CACHE_EEG_POWER_FILENAME, spectral_sample.eeg_power)
        self._atomic_save_array(entry_dir / self.CACHE_FREQUENCIES_FILENAME, spectral_sample.frequencies)
        times_path = entry_dir / self.CACHE_TIMES_FILENAME
        if spectral_sample.times is None:
            times_path.unlink(missing_ok=True)
        else:
            self._atomic_save_array(times_path, spectral_sample.times)
        self._atomic_save_json(
            entry_dir / self.CACHE_MANIFEST_FILENAME,
            self._build_manifest(spectral_sample),
        )

    def _build_manifest(self, spectral_sample: SpectralSample) -> dict[str, Any]:
        sample = spectral_sample.sample
        return {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "transform_version": self.TRANSFORM_VERSION,
            "transform_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "config_hash": self.config_hash,
            "config": self.config.model_dump(mode="json"),
            "key": {
                "subject_id": sample.subject_id,
                "trial_number": sample.trial_number,
                "block_index": sample.block_index,
            },
            "dataset_step_type": self.source_dataset.dataset_step_type,
            "method": self.METHOD,
            "scaling": spectral_sample.scaling,
            "source_dtype": self.source_dtype.name,
            "sources": self.source_dataset._sample_source_signatures(sample),
            "arrays": {
                "eeg_power": self._array_manifest(spectral_sample.eeg_power),
                "frequencies": self._array_manifest(spectral_sample.frequencies),
                "times": None if spectral_sample.times is None else self._array_manifest(spectral_sample.times),
            },
            "source_sfreq": spectral_sample.source_sfreq,
            "analysis_sfreq": spectral_sample.analysis_sfreq,
            "eeg_channels": list(spectral_sample.eeg_channels),
        }

    def _manifest_matches_loaded_sample(self, manifest: dict[str, Any], loaded: LoadedSample) -> bool:
        expected_key = {
            "subject_id": loaded.sample.subject_id,
            "trial_number": loaded.sample.trial_number,
            "block_index": loaded.sample.block_index,
        }
        return (
            manifest.get("schema_version") == self.CACHE_SCHEMA_VERSION
            and manifest.get("transform_version") == self.TRANSFORM_VERSION
            and manifest.get("transform_class") == f"{type(self).__module__}.{type(self).__qualname__}"
            and manifest.get("config_hash") == self.config_hash
            and manifest.get("config") == self.config.model_dump(mode="json")
            and manifest.get("key") == expected_key
            and manifest.get("dataset_step_type") == self.source_dataset.dataset_step_type
            and manifest.get("method") == self.METHOD
            and manifest.get("scaling") == self.config.scaling
            and manifest.get("source_dtype") == self.source_dtype.name
            and manifest.get("sources") == self.source_dataset._sample_source_signatures(loaded.sample)
            and manifest.get("source_sfreq") == loaded.sfreq
            and manifest.get("analysis_sfreq") == self.config.analysis_sfreq
            and manifest.get("eeg_channels") == list(loaded.eeg_channels)
        )

    def _manifest_matches_arrays(
        self,
        manifest: dict[str, Any],
        *,
        eeg_power: np.ndarray,
        frequencies: np.ndarray,
        times: np.ndarray | None,
    ) -> bool:
        arrays = manifest.get("arrays")
        if not isinstance(arrays, dict):
            return False
        expected_dtype = np.dtype(self.config.dtype)
        return (
            arrays.get("eeg_power") == self._array_manifest(eeg_power)
            and arrays.get("frequencies") == self._array_manifest(frequencies)
            and arrays.get("times") == (None if times is None else self._array_manifest(times))
            and eeg_power.dtype == expected_dtype
            and frequencies.dtype == expected_dtype
            and (times is None or times.dtype == expected_dtype)
        )

    @staticmethod
    def _array_manifest(array: np.ndarray) -> dict[str, Any]:
        return {"shape": list(array.shape), "dtype": array.dtype.name}

    @staticmethod
    def _atomic_save_array(path: Path, array: np.ndarray) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as temporary_file:
                temporary_path = Path(temporary_file.name)
                np.save(temporary_file, array, allow_pickle=False)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                suffix=".json",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


class FFTDataset(PreprocessedDataset):
    METHOD = "fft"
    CONFIG_TYPE = FFTConfig

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        return compute_fft_psd(
            loaded.eeg,
            source_sfreq=loaded.sfreq,
            config=self.config,
        )


class MorletDataset(PreprocessedDataset):
    METHOD = "morlet"
    CONFIG_TYPE = MorletConfig

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        return compute_morlet_power(
            loaded.eeg,
            source_sfreq=loaded.sfreq,
            config=self.config,
        )


class SuperletDataset(PreprocessedDataset):
    METHOD = "superlet"
    CONFIG_TYPE = SuperletConfig

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        return compute_superlet_power(
            loaded.eeg,
            source_sfreq=loaded.sfreq,
            config=self.config,
        )


class STFTDataset(PreprocessedDataset):
    METHOD = "stft"
    CONFIG_TYPE = STFTConfig

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        return compute_stft_psd(
            loaded.eeg,
            source_sfreq=loaded.sfreq,
            config=self.config,
        )
