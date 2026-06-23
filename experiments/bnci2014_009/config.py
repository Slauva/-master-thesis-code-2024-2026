from pathlib import Path
from typing import Annotated, Literal, Self

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from preprocessors.config import PreprocessingMethod

BNCI009_LABELS: tuple[str, str] = ("Target", "NonTarget")
BNCI009_SUBJECTS: tuple[int, ...] = tuple(range(1, 11))


class BNCI009DatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_code: Literal["BNCI2014-009"] = "BNCI2014-009"
    subjects: tuple[Annotated[int, Field(ge=1)], ...] = BNCI009_SUBJECTS
    labels: tuple[str, ...] = BNCI009_LABELS
    epoch_start_seconds: Annotated[float, Field(ge=0.0)] = 0.0
    epoch_end_seconds: Annotated[float, Field(gt=0.0)] = 0.8
    source_sfreq: Annotated[float, Field(gt=0.0)] = 256.0
    n_classes: Literal[2] = 2
    dtype: Literal["float32", "float64"] = "float32"
    moabb_filter_low_hz: Annotated[float, Field(ge=0.0)] = 1.0
    moabb_filter_high_hz: Annotated[float, Field(gt=0.0)] = 24.0

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        if not self.subjects:
            raise ValueError("At least one subject is required")
        if len(set(self.subjects)) != len(self.subjects):
            raise ValueError("Subjects must be unique")
        if len(self.labels) != self.n_classes:
            raise ValueError("Label count must match n_classes")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Labels must be unique")
        if set(self.labels) != set(BNCI009_LABELS):
            raise ValueError(f"BNCI2014_009 labels must be exactly {BNCI009_LABELS}")
        if self.epoch_end_seconds <= self.epoch_start_seconds:
            raise ValueError("epoch_end_seconds must be greater than epoch_start_seconds")
        if self.moabb_filter_high_hz <= self.moabb_filter_low_hz:
            raise ValueError("moabb_filter_high_hz must be greater than moabb_filter_low_hz")
        return self


class BNCI009SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    primary_protocol: Literal["leave-one-subject-out"] = "leave-one-subject-out"
    group_by: Literal["subject"] = "subject"
    require_all_classes_in_train: bool = True
    require_all_classes_in_test: bool = True


BNCI009ClassicalModelId = Literal[
    "dummy-prior",
    "erp-lda",
    "erp-logreg",
    "erp-linear-svm",
    "erp-ridge",
    "xdawn-tangent-lda",
    "xdawn-tangent-logreg",
]

BNCI009RawTorchArchitecture = Literal[
    "raw-cnn",
    "eegnet",
    "deep-convnet",
    "shallow-convnet",
]
BNCI009SpectralTorchArchitecture = Literal["eegnet", "deep-convnet", "shallow-convnet"]


class BNCI009ClassicalBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Literal["classical-sweep"] = "classical-sweep"
    variants: tuple[BNCI009ClassicalModelId, ...] = (
        "dummy-prior",
        "erp-lda",
        "erp-logreg",
        "erp-linear-svm",
        "erp-ridge",
        "xdawn-tangent-lda",
        "xdawn-tangent-logreg",
    )
    erp_waveform_stride: Annotated[int, Field(ge=1)] = 4
    xdawn_n_filters: Annotated[int, Field(ge=1)] = 2
    xdawn_estimator: Literal["scm", "lwf", "oas"] = "oas"
    tangent_metric: Literal["riemann", "logeuclid"] = "riemann"
    logistic_c: Annotated[float, Field(gt=0.0)] = 1.0
    logistic_max_iter: Annotated[int, Field(ge=1)] = 1000
    svm_alpha: Annotated[float, Field(gt=0.0)] = 1.0e-4
    svm_max_iter: Annotated[int, Field(ge=1)] = 1000
    svm_tol: Annotated[float, Field(gt=0.0)] = 1.0e-3
    seed: int = 42

    @model_validator(mode="after")
    def validate_classical_benchmark(self) -> Self:
        if not self.variants:
            raise ValueError("At least one classical benchmark variant is required")
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("Classical benchmark variants must be unique")
        return self


class BNCI009RawTorchBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Literal["raw-erp-torch"] = "raw-erp-torch"
    architectures: tuple[BNCI009RawTorchArchitecture, ...] = (
        "raw-cnn",
        "eegnet",
        "deep-convnet",
        "shallow-convnet",
    )
    seed: int = 42
    max_epochs: Annotated[int, Field(ge=1)] = 8
    patience: Annotated[int, Field(ge=1)] = 3
    batch_size: Annotated[int, Field(ge=1)] = 512
    learning_rate: Annotated[float, Field(gt=0.0)] = 1.0e-3
    weight_decay: Annotated[float, Field(ge=0.0)] = 1.0e-4
    dropout_rate: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.5
    hidden_channels: Annotated[int, Field(ge=1)] = 16
    validation_strategy: Literal["lowest-train-subject"] = "lowest-train-subject"
    class_weighting: Literal["balanced"] | None = "balanced"
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @model_validator(mode="after")
    def validate_raw_torch(self) -> Self:
        if not self.architectures:
            raise ValueError("At least one raw Torch architecture is required")
        if len(set(self.architectures)) != len(self.architectures):
            raise ValueError("Raw Torch architectures must be unique")
        return self


class BNCI009SpectralTorchBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Literal["spectral-torch"] = "spectral-torch"
    architectures: tuple[BNCI009SpectralTorchArchitecture, ...] = (
        "eegnet",
        "deep-convnet",
        "shallow-convnet",
    )
    spectral_methods: tuple[Literal["fft"], ...] = ("fft",)
    deferred_methods: tuple[PreprocessingMethod, ...] = ("morlet", "superlet", "stft")
    seed: int = 42
    max_epochs: Annotated[int, Field(ge=1)] = 8
    patience: Annotated[int, Field(ge=1)] = 3
    batch_size: Annotated[int, Field(ge=1)] = 512
    learning_rate: Annotated[float, Field(gt=0.0)] = 1.0e-3
    weight_decay: Annotated[float, Field(ge=0.0)] = 1.0e-4
    dropout_rate: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.5
    validation_strategy: Literal["lowest-train-subject"] = "lowest-train-subject"
    class_weighting: Literal["balanced"] | None = "balanced"
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @model_validator(mode="after")
    def validate_spectral_torch(self) -> Self:
        if not self.architectures:
            raise ValueError("At least one spectral Torch architecture is required")
        if len(set(self.architectures)) != len(self.architectures):
            raise ValueError("Spectral Torch architectures must be unique")
        if not self.spectral_methods:
            raise ValueError("At least one spectral preprocessing method is required")
        if len(set(self.spectral_methods)) != len(self.spectral_methods):
            raise ValueError("Spectral preprocessing methods must be unique")
        if set(self.spectral_methods) & set(self.deferred_methods):
            raise ValueError("A spectral method cannot be both active and deferred")
        return self


class BNCI009ArtifactConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path = Path("artifacts/experiments/bnci2014_009")
    schema_version: Literal[1] = 1
    overwrite: bool = False


class BNCI2014009Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: BNCI009DatasetConfig = BNCI009DatasetConfig()
    split: BNCI009SplitConfig = BNCI009SplitConfig()
    classical: BNCI009ClassicalBenchmarkConfig = BNCI009ClassicalBenchmarkConfig()
    raw_torch: BNCI009RawTorchBenchmarkConfig = BNCI009RawTorchBenchmarkConfig()
    spectral_torch: BNCI009SpectralTorchBenchmarkConfig = BNCI009SpectralTorchBenchmarkConfig()
    artifacts: BNCI009ArtifactConfig = BNCI009ArtifactConfig()

    @model_validator(mode="after")
    def validate_artifact_policy(self) -> Self:
        if self.artifacts.overwrite:
            raise ValueError("BNCI2014_009 artifacts are immutable by default; overwrite is not supported")
        return self


def load_bnci009_config(
    config_path: Path | None = None,
    *,
    overrides: dict[str, object] | None = None,
) -> BNCI2014009Config:
    resolved_path = config_path or Path("confs/experiments/bnci2014_009.yaml")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"BNCI2014_009 configuration does not exist: {resolved_path}")

    configs: list[object] = [OmegaConf.load(resolved_path)]
    if overrides:
        configs.append(OmegaConf.create(overrides))
    merged = OmegaConf.merge(*configs)
    payload = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved BNCI2014_009 configuration must be a mapping")
    return BNCI2014009Config.model_validate(payload)
