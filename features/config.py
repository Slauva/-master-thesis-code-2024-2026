import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias

import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

FeatureGroup: TypeAlias = Literal["time", "spectral", "spatial", "local_patterns"]
HistogramMode: TypeAlias = Literal["count", "probability"]


class FrequencyBand(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")]
    f_min: Annotated[float, Field(ge=0)]
    f_max: Annotated[float, Field(gt=0)]

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.f_max <= self.f_min:
            raise ValueError("Frequency-band `f_max` must be greater than `f_min`")
        return self


class FeatureExtractionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_sfreq: Annotated[float, Field(gt=0)] = 125.0
    crop_start_seconds: Annotated[float, Field(ge=0)] = 0.5
    crop_end_seconds: Annotated[float, Field(gt=0)] = 15.5
    window_seconds: Annotated[float, Field(gt=0)] | None = None
    window_stride_seconds: Annotated[float, Field(gt=0)] | None = None
    dtype: Literal["float32", "float64"] = "float32"
    feature_groups: tuple[FeatureGroup, ...] = ("time", "spectral", "spatial", "local_patterns")
    frequency_bands: tuple[FrequencyBand, ...]
    histogram_mode: HistogramMode = "probability"
    local_pattern_neighbors: Annotated[int, Field(ge=2, le=16)] = 8
    transform_eog: Literal[False] = False
    filter_hz: None = None
    notch_hz: None = None
    reference: None = None
    normalization: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        crop_duration = self.crop_end_seconds - self.crop_start_seconds
        if crop_duration <= 0:
            raise ValueError("`crop_end_seconds` must be greater than `crop_start_seconds`")
        if not self.feature_groups:
            raise ValueError("At least one feature group must be enabled")
        if len(set(self.feature_groups)) != len(self.feature_groups):
            raise ValueError("Feature groups must be unique")
        if not self.frequency_bands:
            raise ValueError("At least one frequency band must be configured")
        if self.local_pattern_neighbors % 2 != 0:
            raise ValueError("`local_pattern_neighbors` must be even")

        if (self.window_seconds is None) != (self.window_stride_seconds is None):
            raise ValueError("`window_seconds` and `window_stride_seconds` must be set together")
        if self.window_seconds is not None:
            if self.window_seconds > crop_duration:
                raise ValueError("`window_seconds` must not exceed the crop duration")
            _require_integer_samples(
                self.window_seconds,
                self.analysis_sfreq,
                field_name="window_seconds",
            )
            _require_integer_samples(
                self.window_stride_seconds,
                self.analysis_sfreq,
                field_name="window_stride_seconds",
            )

        _require_integer_samples(
            crop_duration,
            self.analysis_sfreq,
            field_name="crop duration",
        )
        self._validate_frequency_bands()
        return self

    def _validate_frequency_bands(self) -> None:
        names = [band.name for band in self.frequency_bands]
        if len(set(names)) != len(names):
            raise ValueError("Frequency-band names must be unique")

        nyquist = self.analysis_sfreq / 2.0
        previous_max: float | None = None
        for band in self.frequency_bands:
            if band.f_max > nyquist:
                raise ValueError(
                    f"Frequency band {band.name!r} exceeds the analysis Nyquist frequency ({nyquist:g} Hz)"
                )
            if previous_max is not None and band.f_min < previous_max:
                raise ValueError("Frequency bands must be ordered and non-overlapping")
            previous_max = band.f_max


def load_feature_config(
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> FeatureExtractionConfig:
    resolved_path = Path(config_path) if config_path is not None else _default_config_path()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Feature configuration does not exist: {resolved_path}")

    configs: list[Any] = [OmegaConf.load(resolved_path)]
    if overrides:
        configs.append(OmegaConf.create(overrides))
    merged = OmegaConf.merge(*configs)
    payload = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved feature configuration must be a mapping")
    return FeatureExtractionConfig.model_validate(payload)


def build_feature_config_hash(
    config: FeatureExtractionConfig,
    *,
    cache_schema_version: int = 1,
    extractor_version: int = 1,
) -> str:
    if cache_schema_version < 1 or extractor_version < 1:
        raise ValueError("Cache schema and extractor versions must be positive")
    payload = {
        "cache_schema_version": cache_schema_version,
        "extractor_version": extractor_version,
        "config": config.model_dump(mode="json"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def _require_integer_samples(seconds: float | None, sfreq: float, *, field_name: str) -> int:
    if seconds is None:
        raise ValueError(f"`{field_name}` must not be None")
    exact_samples = seconds * sfreq
    samples = round(exact_samples)
    if not np.isclose(exact_samples, samples, rtol=0.0, atol=1e-12):
        raise ValueError(f"`{field_name} * analysis_sfreq` must be an integer number of samples")
    return samples


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "confs" / "features" / "default.yaml"
