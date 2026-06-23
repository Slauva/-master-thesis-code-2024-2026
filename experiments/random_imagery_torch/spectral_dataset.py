import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np

from experiments.random_imagery_torch.config import SpectralInputConfig
from experiments.random_imagery_torch.schemas import CropSpectralSample
from preprocessors.config import (
    PreprocessingConfig,
    PreprocessingMethod,
    build_frequency_grid,
    load_preprocessing_config,
)
from preprocessors.fft import compute_fft_psd
from preprocessors.morlet import compute_morlet_power
from preprocessors.schemas import SpectralTransformResult
from preprocessors.stft import compute_stft_psd
from preprocessors.superlet import compute_superlet_power
from utils.datasets.base import SampleKey
from utils.datasets.numpy_dataset import NumpyDataset
from utils.datasets.schemas import LoadedSample, Sample

_TRANSFORMS = {
    "fft": compute_fft_psd,
    "morlet": compute_morlet_power,
    "superlet": compute_superlet_power,
    "stft": compute_stft_psd,
}


class CropSpectralDataset:
    CACHE_SCHEMA_VERSION = 1
    TRANSFORM_VERSION = 1
    CACHE_MANIFEST_FILENAME = "manifest.json"
    CACHE_EEG_POWER_FILENAME = "eeg_power.npy"
    CACHE_FREQUENCIES_FILENAME = "frequencies.npy"
    CACHE_TIMES_FILENAME = "times.npy"

    def __init__(
        self,
        source_dataset: NumpyDataset,
        *,
        method: PreprocessingMethod,
        preprocessing_config: PreprocessingConfig | None = None,
        preprocessing_config_overrides: dict[str, object] | None = None,
        input_config: SpectralInputConfig | None = None,
        cache_policy: Literal["none", "disk"] | None = "disk",
        cache_dir: Path | None = None,
    ) -> None:
        if not isinstance(source_dataset, NumpyDataset):
            raise TypeError("`source_dataset` must be a NumpyDataset")
        if source_dataset.dataset_step_type != "patt":
            raise ValueError("Random-imagery spectral inputs require dataset_step_type='patt'")
        if source_dataset.dataset_pattern_type not in ("geometric", "random", None):
            raise ValueError(
                "Random-imagery spectral inputs require dataset_pattern_type to select "
                "geometric, random, or both"
            )
        if not source_dataset.samples:
            raise TypeError("Random-imagery spectral inputs require at least one labeled sample")
        if preprocessing_config is not None and preprocessing_config_overrides is not None:
            raise ValueError("Pass either `preprocessing_config` or overrides, not both")
        resolved_preprocessing = preprocessing_config or load_preprocessing_config(
            method,
            overrides=preprocessing_config_overrides,
        )
        if resolved_preprocessing.method != method:
            raise ValueError("Preprocessing method and configuration disagree")
        if cache_policy not in (None, "none", "disk"):
            raise ValueError(f"Unsupported crop spectral cache policy: {cache_policy!r}")

        self.source_dataset = source_dataset
        self.method = method
        self.preprocessing_config = resolved_preprocessing
        self.input_config = input_config or SpectralInputConfig()
        self.cache_policy = cache_policy
        self.config_hash = self._build_config_hash()
        self.cache_dir = self._resolve_cache_dir(cache_dir) if self._uses_disk_cache else None

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, key: int | SampleKey) -> CropSpectralSample:
        sample = self.source_dataset._get_sample(key)
        if self._uses_disk_cache:
            cached = self._load_disk_cache(sample)
            if cached is not None:
                return cached

        loaded = self.source_dataset[key]
        if loaded.sample != sample:
            raise ValueError("Loaded source metadata does not match the requested sample")
        cropped = self._crop_loaded_sample(loaded)
        transformed = self._transform(cropped)
        spectral = self._build_sample(cropped, transformed)
        if self._uses_disk_cache:
            self._write_disk_cache(spectral)
        return spectral

    def __iter__(self) -> Iterator[CropSpectralSample]:
        for index in range(len(self)):
            yield self[index]

    @property
    def samples(self) -> tuple[Sample, ...]:
        return self.source_dataset.samples

    @property
    def sample_keys(self) -> tuple[SampleKey, ...]:
        return tuple(
            (sample.subject_id, sample.trial_number, sample.block_index)
            for sample in self.samples
        )

    @property
    def _uses_disk_cache(self) -> bool:
        return self.cache_policy == "disk"

    def get_cache_entry_path(self, key: int | SampleKey) -> Path:
        if self.cache_dir is None:
            raise ValueError("Crop spectral disk cache is disabled")
        sample = self.source_dataset._get_sample(key)
        return (
            self.cache_dir
            / f"S_{sample.subject_id}"
            / f"Trial_{sample.trial_number}"
            / f"Block_{sample.block_index}"
        )

    def _crop_loaded_sample(self, loaded: LoadedSample) -> LoadedSample:
        if loaded.eeg.ndim != 2 or not np.isfinite(loaded.eeg).all():
            raise ValueError("Source EEG must be a finite array with shape (channel, time)")
        source_slice = self.input_config.source_slice(
            loaded.sfreq,
            n_times=loaded.eeg.shape[-1],
        )
        return LoadedSample(
            sample=loaded.sample,
            eeg=loaded.eeg[:, source_slice],
            eog=np.empty((0, 0), dtype=loaded.eeg.dtype),
            sfreq=loaded.sfreq,
            eeg_channels=loaded.eeg_channels,
            eog_channels=(),
        )

    def _transform(self, loaded: LoadedSample) -> SpectralTransformResult:
        transform = _TRANSFORMS[self.method]
        return transform(
            loaded.eeg,
            source_sfreq=loaded.sfreq,
            config=self.preprocessing_config,  # type: ignore[arg-type]
        )

    def _build_sample(
        self,
        loaded: LoadedSample,
        transformed: SpectralTransformResult,
    ) -> CropSpectralSample:
        expected_frequencies = build_frequency_grid(self.preprocessing_config)
        if transformed.scaling != self.preprocessing_config.scaling:
            raise ValueError("Transform scaling does not match the preprocessing configuration")
        if not np.isclose(
            transformed.analysis_sfreq,
            self.preprocessing_config.analysis_sfreq,
        ):
            raise ValueError("Transform sampling frequency does not match the configuration")
        if transformed.frequencies.shape != expected_frequencies.shape or not np.allclose(
            transformed.frequencies,
            expected_frequencies,
        ):
            raise ValueError("Transform frequencies do not match the configured grid")

        output_dtype = np.dtype(self.preprocessing_config.dtype)
        times = transformed.times
        if times is not None:
            times = times + self.input_config.crop_start_seconds
        return CropSpectralSample(
            sample=loaded.sample,  # type: ignore[arg-type]
            eeg_power=np.asarray(transformed.eeg_power, dtype=output_dtype),
            frequencies=np.asarray(transformed.frequencies, dtype=output_dtype),
            times=None if times is None else np.asarray(times, dtype=output_dtype),
            eeg_channels=loaded.eeg_channels,
            source_sfreq=loaded.sfreq,
            analysis_sfreq=transformed.analysis_sfreq,
            method=self.method,
            scaling=transformed.scaling,
            crop_bounds_seconds=self.input_config.crop_bounds_seconds,
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
                    / "preprocessed-imagery"
                    / self.source_dataset.dataset_dir.name
                    / self.source_dataset.dataset_step_type
                    / self.method
                    / self.config_hash
                )
        raise ValueError("Could not determine crop spectral cache directory")

    def _build_config_hash(self) -> str:
        payload = json.dumps(
            self._cache_identity(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def _cache_identity(self) -> dict[str, Any]:
        return {
            "cache_schema_version": self.CACHE_SCHEMA_VERSION,
            "transform_version": self.TRANSFORM_VERSION,
            "method": self.method,
            "source_dtype": self.source_dataset.dtype.name,
            "preprocessing_config": self.preprocessing_config.model_dump(mode="json"),
            "input_config": self.input_config.model_dump(mode="json"),
        }

    def _load_disk_cache(self, sample: Sample) -> CropSpectralSample | None:
        entry_dir = self.get_cache_entry_path(
            (sample.subject_id, sample.trial_number, sample.block_index)
        )
        manifest_path = entry_dir / self.CACHE_MANIFEST_FILENAME
        power_path = entry_dir / self.CACHE_EEG_POWER_FILENAME
        frequency_path = entry_dir / self.CACHE_FREQUENCIES_FILENAME
        if not (manifest_path.is_file() and power_path.is_file() and frequency_path.is_file()):
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not self._manifest_matches_sample(manifest, sample):
                return None
            power = np.load(power_path, allow_pickle=False)
            frequencies = np.load(frequency_path, allow_pickle=False)
            times_manifest = manifest["arrays"]["times"]
            times_path = entry_dir / self.CACHE_TIMES_FILENAME
            if times_manifest is not None and not times_path.is_file():
                return None
            times = np.load(times_path, allow_pickle=False) if times_manifest is not None else None
            if not self._manifest_matches_arrays(
                manifest,
                eeg_power=power,
                frequencies=frequencies,
                times=times,
            ):
                return None
            return CropSpectralSample(
                sample=sample,
                eeg_power=power,
                frequencies=frequencies,
                times=times,
                eeg_channels=tuple(manifest["eeg_channels"]),
                source_sfreq=float(manifest["source_sfreq"]),
                analysis_sfreq=float(manifest["analysis_sfreq"]),
                method=self.method,
                scaling=manifest["scaling"],
                crop_bounds_seconds=tuple(manifest["crop_bounds_seconds"]),
            )
        except (EOFError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write_disk_cache(self, spectral: CropSpectralSample) -> None:
        entry_dir = self.get_cache_entry_path(spectral.sample_key)
        entry_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_save_array(entry_dir / self.CACHE_EEG_POWER_FILENAME, spectral.eeg_power)
        self._atomic_save_array(entry_dir / self.CACHE_FREQUENCIES_FILENAME, spectral.frequencies)
        times_path = entry_dir / self.CACHE_TIMES_FILENAME
        if spectral.times is None:
            times_path.unlink(missing_ok=True)
        else:
            self._atomic_save_array(times_path, spectral.times)
        self._atomic_save_json(
            entry_dir / self.CACHE_MANIFEST_FILENAME,
            self._build_manifest(spectral),
        )

    def _build_manifest(self, spectral: CropSpectralSample) -> dict[str, Any]:
        return {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "transform_version": self.TRANSFORM_VERSION,
            "config_hash": self.config_hash,
            "cache_identity": self._cache_identity(),
            "key": {
                "subject_id": spectral.sample.subject_id,
                "trial_number": spectral.sample.trial_number,
                "block_index": spectral.sample.block_index,
            },
            "method": self.method,
            "scaling": spectral.scaling,
            "source": self.source_dataset._source_signature(spectral.sample.eeg_path),
            "arrays": {
                "eeg_power": self._array_manifest(spectral.eeg_power),
                "frequencies": self._array_manifest(spectral.frequencies),
                "times": None
                if spectral.times is None
                else self._array_manifest(spectral.times),
            },
            "source_sfreq": spectral.source_sfreq,
            "analysis_sfreq": spectral.analysis_sfreq,
            "eeg_channels": list(spectral.eeg_channels),
            "crop_bounds_seconds": list(spectral.crop_bounds_seconds),
            "time_reference": "source_recording_seconds",
        }

    def _manifest_matches_sample(
        self,
        manifest: dict[str, Any],
        sample: Sample,
    ) -> bool:
        return (
            manifest.get("schema_version") == self.CACHE_SCHEMA_VERSION
            and manifest.get("transform_version") == self.TRANSFORM_VERSION
            and manifest.get("config_hash") == self.config_hash
            and manifest.get("cache_identity") == self._cache_identity()
            and manifest.get("key")
            == {
                "subject_id": sample.subject_id,
                "trial_number": sample.trial_number,
                "block_index": sample.block_index,
            }
            and manifest.get("method") == self.method
            and manifest.get("scaling") == self.preprocessing_config.scaling
            and manifest.get("source")
            == self.source_dataset._source_signature(sample.eeg_path)
            and manifest.get("crop_bounds_seconds")
            == list(self.input_config.crop_bounds_seconds)
            and manifest.get("time_reference") == "source_recording_seconds"
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
        expected_dtype = np.dtype(self.preprocessing_config.dtype)
        return isinstance(arrays, dict) and (
            arrays.get("eeg_power") == self._array_manifest(eeg_power)
            and arrays.get("frequencies") == self._array_manifest(frequencies)
            and arrays.get("times")
            == (None if times is None else self._array_manifest(times))
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
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                suffix=".npy",
                delete=False,
            ) as temporary_file:
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
                json.dump(payload, temporary_file, ensure_ascii=True, indent=2)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
