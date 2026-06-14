import json
import os
import tempfile
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import mne
import numpy as np
from numpy.typing import DTypeLike

from utils.datasets.base import DatasetBase, SampleKey
from utils.datasets.schemas import LoadedSample, Sample


class NumpyDataset(DatasetBase):
    CACHE_SCHEMA_VERSION = 1
    CACHE_MANIFEST_FILENAME = "manifest.json"
    CACHE_EEG_FILENAME = "eeg.npy"
    CACHE_EOG_FILENAME = "eog.npy"

    def __init__(
        self,
        dataset_dir: Path,
        *,
        dataset_step_type: Literal["exec", "patt"] = "exec",
        dataset_pattern_type: Literal["geometric", "random"] | None = None,
        dtype: DTypeLike = np.float32,
        cache_policy: Literal["none", "memory", "disk", "both"] | None = "disk",
        cache_dir: Path | None = None,
        memory_cache_bytes: int = 1 << 30,
        preload: bool = False,
        exclude_samples: dict[str, list[str]] | None = None,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            dataset_step_type=dataset_step_type,
            exclude_samples=exclude_samples,
        )

        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f"`dtype` must be float32 or float64, got {self.dtype}")

        if cache_policy not in (None, "none", "memory", "disk", "both"):
            raise ValueError(f"Unsupported cache policy: {cache_policy!r}")
        if isinstance(memory_cache_bytes, bool) or not isinstance(memory_cache_bytes, int) or memory_cache_bytes < 0:
            raise ValueError("`memory_cache_bytes` must be a non-negative integer")
        self.cache_policy = cache_policy
        self.cache_dir = self._resolve_cache_dir(cache_dir) if self._uses_disk_cache else cache_dir
        self.memory_cache_bytes = memory_cache_bytes
        self._memory_cache: OrderedDict[
            SampleKey,
            tuple[LoadedSample, dict[str, dict[str, int | str]]],
        ] = OrderedDict()
        self._memory_cache_current_bytes = 0
        self.preload = preload

        self.dataset_pattern_type = dataset_pattern_type
        if dataset_pattern_type is not None:
            filtered_samples = tuple(sample for sample in self.samples if sample.type == dataset_pattern_type)
            self._filter_index(filtered_samples)

    def __getitem__(self, key: int | SampleKey) -> LoadedSample:
        sample = self._get_sample(key)
        sample_key = self._sample_key(sample)
        if self._uses_memory_cache:
            cached = self._load_memory_cache(sample_key, sample)
            if cached is not None:
                return cached

        if self._uses_disk_cache:
            cached = self._load_disk_cache(sample)
            if cached is not None:
                if self._uses_memory_cache:
                    self._store_memory_cache(sample_key, cached)
                return cached

        loaded = self._load_sample(sample)
        if self._uses_disk_cache:
            self._write_disk_cache(loaded)
        if self._uses_memory_cache:
            self._store_memory_cache(sample_key, loaded)
        return loaded

    def __iter__(self) -> Iterator[LoadedSample]:
        for index in range(len(self)):
            yield self[index]

    @property
    def _uses_disk_cache(self) -> bool:
        return self.cache_policy in ("disk", "both")

    @property
    def _uses_memory_cache(self) -> bool:
        return self.cache_policy in ("memory", "both")

    @property
    def memory_cache_current_bytes(self) -> int:
        return self._memory_cache_current_bytes

    @property
    def memory_cache_items(self) -> int:
        return len(self._memory_cache)

    @property
    def memory_cache_keys(self) -> tuple[SampleKey, ...]:
        return tuple(self._memory_cache)

    def clear_memory_cache(self) -> None:
        self._memory_cache.clear()
        self._memory_cache_current_bytes = 0

    def _get_sample(self, key: int | SampleKey) -> Sample:
        if isinstance(key, bool):
            raise TypeError("NumpyDataset key must be an integer or a (subject_id, trial_number, block_index) tuple")
        if isinstance(key, int):
            return self.samples[key]
        return super().__getitem__(key)

    def _load_memory_cache(self, key: SampleKey, sample: Sample) -> LoadedSample | None:
        cache_entry = self._memory_cache.get(key)
        if cache_entry is None:
            return None

        loaded, source_signatures = cache_entry
        current_signatures = self._sample_source_signatures(sample)
        if source_signatures != current_signatures:
            self._remove_memory_cache_entry(key)
            return None

        self._memory_cache.move_to_end(key)
        return loaded

    def _store_memory_cache(self, key: SampleKey, loaded: LoadedSample) -> None:
        loaded_bytes = self._loaded_sample_nbytes(loaded)
        if loaded_bytes > self.memory_cache_bytes:
            return

        if key in self._memory_cache:
            self._remove_memory_cache_entry(key)

        while self._memory_cache and self._memory_cache_current_bytes + loaded_bytes > self.memory_cache_bytes:
            oldest_key = next(iter(self._memory_cache))
            self._remove_memory_cache_entry(oldest_key)

        self._memory_cache[key] = (loaded, self._sample_source_signatures(loaded.sample))
        self._memory_cache_current_bytes += loaded_bytes

    def _remove_memory_cache_entry(self, key: SampleKey) -> None:
        loaded, _ = self._memory_cache.pop(key)
        self._memory_cache_current_bytes -= self._loaded_sample_nbytes(loaded)

    def _load_sample(self, sample: Sample) -> LoadedSample:
        eeg_raw = mne.io.read_raw_fif(sample.eeg_path, preload=False, verbose="ERROR")
        eog_raw = mne.io.read_raw_fif(sample.eog_path, preload=False, verbose="ERROR")
        try:
            self._validate_raw_pair(sample, eeg_raw, eog_raw)
            eeg = eeg_raw.get_data().astype(self.dtype, copy=False)
            eog = eog_raw.get_data().astype(self.dtype, copy=False)
            return LoadedSample(
                sample=sample,
                eeg=eeg,
                eog=eog,
                sfreq=float(eeg_raw.info["sfreq"]),
                eeg_channels=tuple(eeg_raw.ch_names),
                eog_channels=tuple(eog_raw.ch_names),
            )
        finally:
            eeg_raw.close()
            eog_raw.close()

    def get_cache_entry_path(self, key: int | SampleKey) -> Path:
        if self.cache_dir is None:
            raise ValueError("Disk cache is disabled for this dataset")
        sample = self._get_sample(key)
        return (
            self.cache_dir
            / f"S_{sample.subject_id}"
            / f"Trial_{sample.trial_number}"
            / f"Block_{sample.block_index}"
        )

    def _resolve_cache_dir(self, cache_dir: Path | None) -> Path:
        if cache_dir is not None:
            return Path(cache_dir)

        for candidate in (self.dataset_dir.resolve(), *self.dataset_dir.resolve().parents):
            if (candidate / "pyproject.toml").is_file():
                return (
                    candidate
                    / "artifacts"
                    / "cache"
                    / self.dataset_dir.name
                    / self.dataset_step_type
                    / self.dtype.name
                )

        raise ValueError("Could not determine default cache directory; pass `cache_dir` explicitly")

    def _load_disk_cache(self, sample: Sample) -> LoadedSample | None:
        entry_dir = self.get_cache_entry_path((sample.subject_id, sample.trial_number, sample.block_index))
        manifest_path = entry_dir / self.CACHE_MANIFEST_FILENAME
        eeg_path = entry_dir / self.CACHE_EEG_FILENAME
        eog_path = entry_dir / self.CACHE_EOG_FILENAME
        if not (manifest_path.is_file() and eeg_path.is_file() and eog_path.is_file()):
            return None

        try:
            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            if not self._manifest_matches_sample(manifest, sample):
                return None

            eeg = np.load(eeg_path, allow_pickle=False)
            eog = np.load(eog_path, allow_pickle=False)
            if not self._manifest_matches_arrays(manifest, eeg=eeg, eog=eog):
                return None

            return LoadedSample(
                sample=sample,
                eeg=eeg,
                eog=eog,
                sfreq=float(manifest["sfreq"]),
                eeg_channels=tuple(manifest["eeg_channels"]),
                eog_channels=tuple(manifest["eog_channels"]),
            )
        except (EOFError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write_disk_cache(self, loaded: LoadedSample) -> None:
        entry_dir = self.get_cache_entry_path(
            (loaded.sample.subject_id, loaded.sample.trial_number, loaded.sample.block_index)
        )
        entry_dir.mkdir(parents=True, exist_ok=True)

        self._atomic_save_array(entry_dir / self.CACHE_EEG_FILENAME, loaded.eeg)
        self._atomic_save_array(entry_dir / self.CACHE_EOG_FILENAME, loaded.eog)
        self._atomic_save_json(
            entry_dir / self.CACHE_MANIFEST_FILENAME,
            self._build_manifest(loaded),
        )

    def _build_manifest(self, loaded: LoadedSample) -> dict[str, Any]:
        sample = loaded.sample
        return {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "key": {
                "subject_id": sample.subject_id,
                "trial_number": sample.trial_number,
                "block_index": sample.block_index,
            },
            "dataset_step_type": self.dataset_step_type,
            "dtype": self.dtype.name,
            "sources": self._sample_source_signatures(sample),
            "arrays": {
                "eeg": {"shape": list(loaded.eeg.shape), "dtype": loaded.eeg.dtype.name},
                "eog": {"shape": list(loaded.eog.shape), "dtype": loaded.eog.dtype.name},
            },
            "sfreq": loaded.sfreq,
            "eeg_channels": list(loaded.eeg_channels),
            "eog_channels": list(loaded.eog_channels),
        }

    def _manifest_matches_sample(self, manifest: dict[str, Any], sample: Sample) -> bool:
        expected_key = {
            "subject_id": sample.subject_id,
            "trial_number": sample.trial_number,
            "block_index": sample.block_index,
        }
        return (
            manifest.get("schema_version") == self.CACHE_SCHEMA_VERSION
            and manifest.get("key") == expected_key
            and manifest.get("dataset_step_type") == self.dataset_step_type
            and manifest.get("dtype") == self.dtype.name
            and manifest.get("sources") == self._sample_source_signatures(sample)
        )

    def _manifest_matches_arrays(self, manifest: dict[str, Any], *, eeg: np.ndarray, eog: np.ndarray) -> bool:
        arrays = manifest.get("arrays")
        if not isinstance(arrays, dict):
            return False
        return (
            arrays.get("eeg") == {"shape": list(eeg.shape), "dtype": eeg.dtype.name}
            and arrays.get("eog") == {"shape": list(eog.shape), "dtype": eog.dtype.name}
            and eeg.dtype == self.dtype
            and eog.dtype == self.dtype
        )

    @staticmethod
    def _source_signature(path: Path) -> dict[str, int | str]:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _sample_source_signatures(self, sample: Sample) -> dict[str, dict[str, int | str]]:
        return {
            "eeg": self._source_signature(sample.eeg_path),
            "eog": self._source_signature(sample.eog_path),
        }

    @staticmethod
    def _sample_key(sample: Sample) -> SampleKey:
        return sample.subject_id, sample.trial_number, sample.block_index

    @staticmethod
    def _loaded_sample_nbytes(loaded: LoadedSample) -> int:
        return loaded.eeg.nbytes + loaded.eog.nbytes

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

    @staticmethod
    def _validate_raw_pair(sample: Sample, eeg_raw: mne.io.BaseRaw, eog_raw: mne.io.BaseRaw) -> None:
        eeg_sfreq = float(eeg_raw.info["sfreq"])
        eog_sfreq = float(eog_raw.info["sfreq"])
        if not np.isclose(eeg_sfreq, eog_sfreq):
            raise ValueError(
                f"EEG/EOG sampling frequencies differ for subject={sample.subject_id}, "
                f"trial={sample.trial_number}, block={sample.block_index}: {eeg_sfreq} != {eog_sfreq}"
            )
        if eeg_raw.n_times != eog_raw.n_times:
            raise ValueError(
                f"EEG/EOG sample counts differ for subject={sample.subject_id}, "
                f"trial={sample.trial_number}, block={sample.block_index}: "
                f"{eeg_raw.n_times} != {eog_raw.n_times}"
            )
