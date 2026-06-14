from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias

import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

PreprocessingMethod: TypeAlias = Literal["fft", "morlet", "superlet", "stft"]
SpectralScaling: TypeAlias = Literal["psd", "wavelet_power"]


class CommonPreprocessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    method: PreprocessingMethod
    scaling: SpectralScaling
    analysis_sfreq: Annotated[float, Field(gt=0)]
    f_min: Annotated[float, Field(gt=0)]
    f_max: Annotated[float, Field(gt=0)]
    frequency_step: Annotated[float, Field(gt=0)]
    dtype: Literal["float32", "float64"] = "float32"
    transform_eog: Literal[False] = False
    filter_hz: None = None
    notch_hz: None = None
    reference: None = None
    normalization: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_frequency_range(self) -> Self:
        if self.f_max <= self.f_min:
            raise ValueError("`f_max` must be greater than `f_min`")
        nyquist = self.analysis_sfreq / 2
        if self.f_max > nyquist:
            raise ValueError(f"`f_max` must not exceed the analysis Nyquist frequency ({nyquist:g} Hz)")
        intervals = (self.f_max - self.f_min) / self.frequency_step
        if not np.isclose(intervals, round(intervals)):
            raise ValueError("The frequency range must be evenly divisible by `frequency_step`")
        return self


class FFTConfig(CommonPreprocessingConfig):
    method: Literal["fft"] = "fft"
    scaling: Literal["psd"] = "psd"
    window: Literal["hann"] = "hann"
    demean: bool = True


class MorletConfig(CommonPreprocessingConfig):
    method: Literal["morlet"] = "morlet"
    scaling: Literal["wavelet_power"] = "wavelet_power"
    zero_mean: Literal[True] = True
    use_fft: bool = True
    n_cycles_divisor: Annotated[float, Field(gt=0)] = 2.0
    n_cycles_min: Annotated[float, Field(gt=0)] = 3.0
    n_cycles_max: Annotated[float, Field(gt=0)] = 10.0
    time_bin_samples: Annotated[int, Field(gt=0)] = 32

    @model_validator(mode="after")
    def validate_cycle_range(self) -> Self:
        if self.n_cycles_max < self.n_cycles_min:
            raise ValueError("`n_cycles_max` must be greater than or equal to `n_cycles_min`")
        return self


class SuperletConfig(CommonPreprocessingConfig):
    method: Literal["superlet"] = "superlet"
    scaling: Literal["wavelet_power"] = "wavelet_power"
    adaptive: Literal[True] = True
    order_min: Annotated[int, Field(ge=1)] = 1
    order_max: Annotated[int, Field(ge=1)] = 10
    c_1: Annotated[int, Field(ge=1)] = 3
    time_bin_samples: Annotated[int, Field(gt=0)] = 32

    @model_validator(mode="after")
    def validate_order_range(self) -> Self:
        if self.order_max < self.order_min:
            raise ValueError("`order_max` must be greater than or equal to `order_min`")
        if self.c_1 * self.order_min < 3:
            raise ValueError("The minimum Superlet cycle count `c_1 * order_min` must be at least 3")
        return self


class STFTConfig(CommonPreprocessingConfig):
    method: Literal["stft"] = "stft"
    scaling: Literal["psd"] = "psd"
    window: Literal["hann"] = "hann"
    window_seconds: Annotated[float, Field(gt=0)] = 2.0
    hop_samples: Annotated[int, Field(gt=0)] = 32
    mfft: Annotated[int, Field(gt=0)] = 250
    fft_mode: Literal["onesided2X"] = "onesided2X"

    @model_validator(mode="after")
    def validate_stft_lengths(self) -> Self:
        window_samples = round(self.window_seconds * self.analysis_sfreq)
        if self.hop_samples > window_samples:
            raise ValueError("`hop_samples` must not exceed the STFT window length")
        if self.mfft < window_samples:
            raise ValueError("`mfft` must be greater than or equal to the STFT window length")
        return self


PreprocessingConfig: TypeAlias = FFTConfig | MorletConfig | SuperletConfig | STFTConfig

_CONFIG_TYPES: dict[PreprocessingMethod, type[PreprocessingConfig]] = {
    "fft": FFTConfig,
    "morlet": MorletConfig,
    "superlet": SuperletConfig,
    "stft": STFTConfig,
}


def load_preprocessing_config(
    method: PreprocessingMethod,
    *,
    config_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> PreprocessingConfig:
    resolved_config_dir = Path(config_dir) if config_dir is not None else _default_config_dir()
    common_path = resolved_config_dir / "common.yaml"
    method_path = resolved_config_dir / f"{method}.yaml"
    for path in (common_path, method_path):
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessing config does not exist: {path}")

    configs: list[Any] = [OmegaConf.load(common_path), OmegaConf.load(method_path)]
    if overrides:
        configs.append(OmegaConf.create(overrides))
    merged = OmegaConf.merge(*configs)
    payload = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise TypeError("Merged preprocessing configuration must be a mapping")
    return _CONFIG_TYPES[method].model_validate(payload)


def build_frequency_grid(config: CommonPreprocessingConfig) -> NDArray[np.float64]:
    count = round((config.f_max - config.f_min) / config.frequency_step) + 1
    frequencies = config.f_min + np.arange(count, dtype=np.float64) * config.frequency_step
    frequencies[-1] = config.f_max
    return frequencies


def _default_config_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "confs" / "preprocessing"
