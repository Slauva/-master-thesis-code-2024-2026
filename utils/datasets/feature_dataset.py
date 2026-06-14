import json
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import DTypeLike

from features.config import FeatureExtractionConfig, build_feature_config_hash, load_feature_config
from features.extractor import extract_feature_set
from features.schemas import FeatureBlock, FeatureSet
from utils.datasets.base import SampleKey, SourceMap
from utils.datasets.numpy_dataset import NumpyDataset
from utils.datasets.schemas import Sample


class FeatureDataset:
    CACHE_SCHEMA_VERSION = 1
    EXTRACTOR_VERSION = 1
    CACHE_MANIFEST_FILENAME = "manifest.json"
    CACHE_WINDOW_BOUNDS_FILENAME = "window_bounds_seconds.npy"
    _BLOCK_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(
        self,
        dataset_dir: Path,
        *,
        dataset_step_type: Literal["exec", "patt"] = "exec",
        dataset_pattern_type: Literal["geometric", "random"] | None = None,
        config: FeatureExtractionConfig | None = None,
        config_path: Path | None = None,
        config_overrides: dict[str, Any] | None = None,
        cache_policy: Literal["none", "disk"] | None = "disk",
        cache_dir: Path | None = None,
        source_dtype: DTypeLike = np.float32,
        source_cache_policy: Literal["none", "memory", "disk", "both"] | None = "disk",
        source_cache_dir: Path | None = None,
        source_memory_cache_bytes: int = 1 << 30,
        exclude_samples: dict[str, list[str]] | None = None,
    ) -> None:
        if config is not None and (config_path is not None or config_overrides is not None):
            raise ValueError("Pass either `config` or config loading options, not both")
        if cache_policy not in (None, "none", "disk"):
            raise ValueError(f"Unsupported feature cache policy: {cache_policy!r}")

        self.config = config or load_feature_config(
            config_path=config_path,
            overrides=config_overrides,
        )
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
        self.config_hash = build_feature_config_hash(
            self.config,
            cache_schema_version=self.CACHE_SCHEMA_VERSION,
            extractor_version=self.EXTRACTOR_VERSION,
        )
        self.cache_dir = self._resolve_cache_dir(cache_dir) if self._uses_disk_cache else None

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, key: int | SampleKey) -> FeatureSet:
        sample = self.source_dataset._get_sample(key)
        if self._uses_disk_cache:
            cached = self._load_disk_cache(sample)
            if cached is not None:
                return cached

        loaded = self.source_dataset[key]
        feature_set = extract_feature_set(loaded, config=self.config)
        if self._uses_disk_cache:
            self._write_disk_cache(feature_set, source_sfreq=loaded.sfreq)
        return feature_set

    def __iter__(self) -> Iterator[FeatureSet]:
        for index in range(len(self)):
            yield self[index]

    @property
    def _uses_disk_cache(self) -> bool:
        return self.cache_policy == "disk"

    @property
    def dataset_step_type(self) -> Literal["exec", "patt"]:
        return self.source_dataset.dataset_step_type

    @property
    def samples(self) -> tuple[Sample, ...]:
        return self.source_dataset.samples

    @property
    def source_map(self) -> SourceMap:
        return self.source_dataset.source_map

    def get_cache_entry_path(self, key: int | SampleKey) -> Path:
        if self.cache_dir is None:
            raise ValueError("Feature disk cache is disabled for this dataset")
        sample = self.source_dataset._get_sample(key)
        return (
            self.cache_dir
            / f"S_{sample.subject_id}"
            / f"Trial_{sample.trial_number}"
            / f"Block_{sample.block_index}"
        )

    def _resolve_cache_dir(self, cache_dir: Path | None) -> Path:
        if cache_dir is not None:
            return (
                Path(cache_dir)
                / self.source_dataset.dataset_dir.name
                / self.dataset_step_type
                / self.source_dtype.name
                / self.config_hash
            )

        dataset_dir = self.source_dataset.dataset_dir.resolve()
        for candidate in (dataset_dir, *dataset_dir.parents):
            if (candidate / "pyproject.toml").is_file():
                return (
                    candidate
                    / "artifacts"
                    / "features"
                    / self.source_dataset.dataset_dir.name
                    / self.dataset_step_type
                    / self.source_dtype.name
                    / self.config_hash
                )
        raise ValueError("Could not determine default feature cache directory; pass `cache_dir` explicitly")

    def _load_disk_cache(self, sample: Sample) -> FeatureSet | None:
        entry_dir = self.get_cache_entry_path(
            (sample.subject_id, sample.trial_number, sample.block_index)
        )
        manifest_path = entry_dir / self.CACHE_MANIFEST_FILENAME
        bounds_path = entry_dir / self.CACHE_WINDOW_BOUNDS_FILENAME
        if not (manifest_path.is_file() and bounds_path.is_file()):
            return None

        try:
            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            if not self._manifest_matches_sample(manifest, sample):
                return None

            window_bounds = np.load(bounds_path, allow_pickle=False)
            if manifest["arrays"]["window_bounds_seconds"] != self._array_manifest(window_bounds):
                return None
            if window_bounds.dtype != np.dtype(np.float64):
                return None

            blocks = tuple(
                self._load_block(entry_dir, block_manifest)
                for block_manifest in manifest["blocks"]
            )
            return FeatureSet(
                sample=sample,
                blocks=blocks,
                window_bounds_seconds=window_bounds,
                eeg_channels=tuple(manifest["eeg_channels"]),
                analysis_sfreq=float(manifest["analysis_sfreq"]),
            )
        except (EOFError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _load_block(self, entry_dir: Path, block_manifest: dict[str, Any]) -> FeatureBlock:
        name = block_manifest["name"]
        if not isinstance(name, str) or self._BLOCK_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError("Invalid cached feature block name")
        expected_filename = f"{name}.npy"
        if block_manifest["filename"] != expected_filename:
            raise ValueError("Cached feature block filename does not match its name")

        values = np.load(entry_dir / expected_filename, allow_pickle=False)
        if block_manifest["array"] != self._array_manifest(values):
            raise ValueError("Cached feature block array metadata does not match")
        if values.dtype != np.dtype(self.config.dtype):
            raise ValueError("Cached feature block dtype does not match configuration")
        return FeatureBlock(
            name=name,
            layout=block_manifest["layout"],
            values=values,
            feature_names=tuple(block_manifest["feature_names"]),
        )

    def _write_disk_cache(self, feature_set: FeatureSet, *, source_sfreq: float) -> None:
        entry_dir = self.get_cache_entry_path(
            (
                feature_set.sample.subject_id,
                feature_set.sample.trial_number,
                feature_set.sample.block_index,
            )
        )
        entry_dir.mkdir(parents=True, exist_ok=True)

        self._atomic_save_array(
            entry_dir / self.CACHE_WINDOW_BOUNDS_FILENAME,
            feature_set.window_bounds_seconds,
        )
        for block in feature_set.blocks:
            if self._BLOCK_NAME_PATTERN.fullmatch(block.name) is None:
                raise ValueError(f"Feature block name is not cache-safe: {block.name!r}")
            self._atomic_save_array(entry_dir / f"{block.name}.npy", block.values)
        self._atomic_save_json(
            entry_dir / self.CACHE_MANIFEST_FILENAME,
            self._build_manifest(feature_set, source_sfreq=source_sfreq),
        )

    def _build_manifest(self, feature_set: FeatureSet, *, source_sfreq: float) -> dict[str, Any]:
        sample = feature_set.sample
        return {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "extractor_version": self.EXTRACTOR_VERSION,
            "extractor_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "config_hash": self.config_hash,
            "config": self.config.model_dump(mode="json"),
            "key": {
                "subject_id": sample.subject_id,
                "trial_number": sample.trial_number,
                "block_index": sample.block_index,
            },
            "dataset_step_type": self.dataset_step_type,
            "source_dtype": self.source_dtype.name,
            "sources": self.source_dataset._sample_source_signatures(sample),
            "source_sfreq": source_sfreq,
            "analysis_sfreq": feature_set.analysis_sfreq,
            "eeg_channels": list(feature_set.eeg_channels),
            "arrays": {
                "window_bounds_seconds": self._array_manifest(feature_set.window_bounds_seconds),
            },
            "blocks": [
                {
                    "name": block.name,
                    "layout": block.layout,
                    "filename": f"{block.name}.npy",
                    "feature_names": list(block.feature_names),
                    "array": self._array_manifest(block.values),
                }
                for block in feature_set.blocks
            ],
        }

    def _manifest_matches_sample(self, manifest: dict[str, Any], sample: Sample) -> bool:
        expected_key = {
            "subject_id": sample.subject_id,
            "trial_number": sample.trial_number,
            "block_index": sample.block_index,
        }
        blocks = manifest.get("blocks")
        return (
            manifest.get("schema_version") == self.CACHE_SCHEMA_VERSION
            and manifest.get("extractor_version") == self.EXTRACTOR_VERSION
            and manifest.get("extractor_class") == f"{type(self).__module__}.{type(self).__qualname__}"
            and manifest.get("config_hash") == self.config_hash
            and manifest.get("config") == self.config.model_dump(mode="json")
            and manifest.get("key") == expected_key
            and manifest.get("dataset_step_type") == self.dataset_step_type
            and manifest.get("source_dtype") == self.source_dtype.name
            and manifest.get("sources") == self.source_dataset._sample_source_signatures(sample)
            and isinstance(manifest.get("source_sfreq"), int | float)
            and manifest.get("analysis_sfreq") == self.config.analysis_sfreq
            and isinstance(manifest.get("eeg_channels"), list)
            and isinstance(manifest.get("arrays"), dict)
            and isinstance(blocks, list)
            and len(blocks) > 0
            and [block.get("name") for block in blocks] == self._expected_block_names()
        )

    def _expected_block_names(self) -> list[str]:
        names: list[str] = []
        for group in self.config.feature_groups:
            if group in ("time", "spectral"):
                names.append(group)
            elif group == "spatial":
                names.extend(("covariance", "correlation", "log_covariance"))
            elif group == "local_patterns":
                names.extend(("lndp", "lgp", "lbp"))
        return names

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
