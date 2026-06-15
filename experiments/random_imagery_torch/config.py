import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.random_imagery.config import (
    ArtifactConfig,
    DatasetSelectionConfig,
    SubjectSplitConfig,
)
from preprocessors.config import PreprocessingMethod


class SpectralInputConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    crop_start_seconds: Annotated[float, Field(ge=0)] = 0.5
    crop_end_seconds: Annotated[float, Field(gt=0)] = 15.5
    log_epsilon: Annotated[float, Field(gt=0)] = 1e-12
    std_epsilon: Annotated[float, Field(gt=0)] = 1e-6

    @model_validator(mode="after")
    def validate_crop(self) -> Self:
        if self.crop_end_seconds <= self.crop_start_seconds:
            raise ValueError("`crop_end_seconds` must be greater than `crop_start_seconds`")
        return self

    def source_slice(self, source_sfreq: float, *, n_times: int) -> slice:
        if not np.isfinite(source_sfreq) or source_sfreq <= 0:
            raise ValueError("Source sampling frequency must be finite and positive")
        if isinstance(n_times, bool) or not isinstance(n_times, int) or n_times < 1:
            raise ValueError("Source sample count must be a positive integer")

        start = _seconds_to_samples(
            self.crop_start_seconds,
            source_sfreq,
            name="crop_start_seconds",
        )
        stop = _seconds_to_samples(
            self.crop_end_seconds,
            source_sfreq,
            name="crop_end_seconds",
        )
        if stop > n_times:
            available_seconds = n_times / source_sfreq
            raise ValueError(
                f"Configured crop ends at {self.crop_end_seconds:g} s, but the signal contains "
                f"only {available_seconds:g} s"
            )
        return slice(start, stop)

    @property
    def crop_bounds_seconds(self) -> tuple[float, float]:
        return self.crop_start_seconds, self.crop_end_seconds


class TorchTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: Literal["eegnet", "deep-convnet", "shallow-convnet"]
    method: PreprocessingMethod
    dropout_rate: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.5
    learning_rate: Annotated[float, Field(gt=0.0)] = 1e-3
    weight_decay: Annotated[float, Field(ge=0.0)] = 1e-4
    batch_size: Annotated[int, Field(ge=1)] = 16
    maximum_epochs: Annotated[int, Field(ge=1)] = 300
    validation_folds: Literal[3] = 3
    selection_seed: int = 42
    early_stopping_patience: Annotated[int, Field(ge=1)] = 30
    early_stopping_min_delta: Annotated[float, Field(ge=0.0)] = 1e-4
    gradient_clip_norm: Annotated[float, Field(gt=0.0)] = 1.0
    final_seeds: tuple[int, int, int] = (42, 43, 44)
    prediction_threshold: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    num_workers: Literal[0] = 0
    use_amp: Literal[False] = False
    deterministic: Literal[True] = True
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @model_validator(mode="after")
    def validate_training_contract(self) -> Self:
        if len(set(self.final_seeds)) != len(self.final_seeds):
            raise ValueError("Final ensemble seeds must be unique")
        return self

    @property
    def model_id(self) -> str:
        return f"{self.architecture}-{self.method}-multilabel"


class TorchExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    dataset: DatasetSelectionConfig = DatasetSelectionConfig()
    split: SubjectSplitConfig = SubjectSplitConfig()
    spectral_input: SpectralInputConfig = SpectralInputConfig()
    preprocessing_overrides: dict[str, Any] = Field(default_factory=dict)
    training: TorchTrainingConfig
    bootstrap_iterations: Annotated[int, Field(ge=1)] = 2000
    random_state: int = 42
    artifacts: ArtifactConfig = ArtifactConfig(
        root=Path("artifacts/experiments/random-imagery-torch"),
        schema_version=1,
    )

    @model_validator(mode="after")
    def validate_model_id(self) -> Self:
        if self.model_id != self.training.model_id:
            raise ValueError(
                f"Configuration model_id {self.model_id!r} does not match "
                f"training model_id {self.training.model_id!r}"
            )
        if self.artifacts.schema_version != 1:
            raise ValueError("Torch random-imagery artifacts require schema_version=1")
        if self.artifacts.overwrite:
            raise ValueError("Torch random-imagery runs are immutable; overwrite is not supported")
        return self


PRIMARY_TORCH_MODEL_IDS: tuple[str, ...] = tuple(
    f"{architecture}-{method}-multilabel"
    for architecture in ("eegnet", "deep-convnet", "shallow-convnet")
    for method in ("fft", "morlet", "superlet", "stft")
)


def parse_torch_model_id(model_id: str) -> tuple[str, PreprocessingMethod]:
    suffix = "-multilabel"
    if not model_id.endswith(suffix):
        raise ValueError(f"Unsupported Torch model id: {model_id!r}")
    body = model_id.removesuffix(suffix)
    for method in ("superlet", "morlet", "stft", "fft"):
        method_suffix = f"-{method}"
        if body.endswith(method_suffix):
            architecture = body.removesuffix(method_suffix)
            if architecture not in {"eegnet", "deep-convnet", "shallow-convnet"}:
                raise ValueError(f"Unsupported Torch architecture in model id: {model_id!r}")
            return architecture, method  # type: ignore[return-value]
    raise ValueError(f"Unsupported Torch preprocessing method in model id: {model_id!r}")


def build_torch_run_hash(
    config: TorchExperimentConfig,
    *,
    protocol: str,
    direction: str,
    experiment_version: int = 1,
) -> str:
    expected_directions = {
        "cross-subject": {"cross-subject"},
        "within-subject": {"trial-1-to-trial-2", "trial-2-to-trial-1"},
    }
    if protocol not in expected_directions:
        raise ValueError(f"Unsupported evaluation protocol: {protocol!r}")
    if direction not in expected_directions[protocol]:
        raise ValueError(f"Direction {direction!r} does not belong to protocol {protocol!r}")
    payload = {
        "experiment_version": experiment_version,
        "artifact_schema_version": config.artifacts.schema_version,
        "model_id": config.model_id,
        "protocol": protocol,
        "direction": direction,
        "config": config.model_dump(mode="json"),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def load_torch_config(
    model_id: str,
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> TorchExperimentConfig:
    architecture, method = parse_torch_model_id(model_id)
    resolved_path = (
        Path(config_path)
        if config_path is not None
        else Path("confs/experiments/random_imagery_torch.yaml")
    )
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Torch random-imagery configuration does not exist: {resolved_path}")
    configs: list[Any] = [OmegaConf.load(resolved_path)]
    configs.append(
        OmegaConf.create(
            {
                "model_id": model_id,
                "training": {
                    "architecture": architecture,
                    "method": method,
                },
            }
        )
    )
    if overrides:
        configs.append(OmegaConf.create(overrides))
    merged = OmegaConf.merge(*configs)
    payload = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved Torch random-imagery configuration must be a mapping")
    config = TorchExperimentConfig.model_validate(payload)
    if config.model_id != model_id:
        raise ValueError(f"Configuration model {config.model_id!r} does not match {model_id!r}")
    return config


def _seconds_to_samples(seconds: float, sfreq: float, *, name: str) -> int:
    exact_samples = seconds * sfreq
    samples = round(exact_samples)
    if not np.isclose(exact_samples, samples, rtol=0.0, atol=1e-12):
        raise ValueError(f"`{name}` does not resolve to an integer sample at {sfreq:g} Hz")
    return samples
