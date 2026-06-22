from pathlib import Path
from typing import Annotated, Literal, Self

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from features.config import FeatureGroup
from preprocessors.config import PreprocessingMethod

BNCI_LABELS: tuple[str, str, str, str] = (
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
)
BNCI_SUBJECTS: tuple[int, ...] = tuple(range(1, 10))


class BNCIDatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_code: Literal["BNCI2014-001"] = "BNCI2014-001"
    subjects: tuple[Annotated[int, Field(ge=1)], ...] = BNCI_SUBJECTS
    labels: tuple[str, ...] = BNCI_LABELS
    epoch_start_seconds: Annotated[float, Field(ge=0.0)] = 2.0
    epoch_end_seconds: Annotated[float, Field(gt=0.0)] = 6.0
    source_sfreq: Annotated[float, Field(gt=0.0)] = 250.0
    n_classes: Literal[4] = 4
    dtype: Literal["float32", "float64"] = "float32"

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
        if self.epoch_end_seconds <= self.epoch_start_seconds:
            raise ValueError("epoch_end_seconds must be greater than epoch_start_seconds")
        return self


class BNCISplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    primary_protocol: Literal["leave-one-subject-out"] = "leave-one-subject-out"
    group_by: Literal["subject"] = "subject"
    require_all_classes_in_train: bool = True
    require_all_classes_in_test: bool = True


class BNCICSPBaselineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Literal["csp-lda"] = "csp-lda"
    n_components: Annotated[int, Field(ge=1)] = 8
    reg: float | Literal["ledoit_wolf", "oas"] | None = None
    log: bool | None = True
    cov_est: Literal["concat", "epoch"] = "concat"
    norm_trace: bool = False
    component_order: Literal["mutual_info", "alternate"] = "mutual_info"
    lda_solver: Literal["svd", "lsqr", "eigen"] = "svd"
    lda_shrinkage: float | Literal["auto"] | None = None

    @model_validator(mode="after")
    def validate_lda(self) -> Self:
        if self.lda_solver == "svd" and self.lda_shrinkage is not None:
            raise ValueError("lda_shrinkage is only supported for lsqr or eigen solvers")
        return self


class BNCIProjectFeatureBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Literal["feature-logreg"] = "feature-logreg"
    feature_groups: tuple[FeatureGroup, ...] = ("time", "spectral")
    window_seconds: Annotated[float, Field(gt=0.0)] | None = None
    window_stride_seconds: Annotated[float, Field(gt=0.0)] | None = None
    logistic_c: Annotated[float, Field(gt=0.0)] = 1.0
    logistic_solver: Literal["lbfgs", "liblinear"] = "liblinear"
    logistic_class_weight: Literal["balanced"] | None = "balanced"
    logistic_max_iter: Annotated[int, Field(ge=1)] = 1000
    standardize: Literal[True] = True

    @model_validator(mode="after")
    def validate_feature_benchmark(self) -> Self:
        if not self.feature_groups:
            raise ValueError("At least one feature group is required")
        if len(set(self.feature_groups)) != len(self.feature_groups):
            raise ValueError("Feature groups must be unique")
        if (self.window_seconds is None) != (self.window_stride_seconds is None):
            raise ValueError("window_seconds and window_stride_seconds must be set together")
        return self


class BNCITorchPilotConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Literal["fft-cnn-pilot"] = "fft-cnn-pilot"
    spectral_method: Literal["fft"] = "fft"
    seed: int = 42
    max_epochs: Annotated[int, Field(ge=1)] = 12
    patience: Annotated[int, Field(ge=1)] = 4
    batch_size: Annotated[int, Field(ge=1)] = 256
    learning_rate: Annotated[float, Field(gt=0.0)] = 1.0e-3
    weight_decay: Annotated[float, Field(ge=0.0)] = 1.0e-4
    hidden_channels: Annotated[int, Field(ge=1)] = 16
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.25
    validation_strategy: Literal["lowest-train-subject"] = "lowest-train-subject"
    device: Literal["auto", "cpu", "cuda"] = "auto"


class BNCITorchFullBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Literal["torch-full"] = "torch-full"
    architectures: tuple[Literal["eegnet", "deep-convnet", "shallow-convnet"], ...] = (
        "eegnet",
        "deep-convnet",
        "shallow-convnet",
    )
    spectral_methods: tuple[PreprocessingMethod, ...] = ("fft", "morlet", "stft", "superlet")
    seed: int = 42
    max_epochs: Annotated[int, Field(ge=1)] = 12
    patience: Annotated[int, Field(ge=1)] = 4
    batch_size: Annotated[int, Field(ge=1)] = 256
    learning_rate: Annotated[float, Field(gt=0.0)] = 1.0e-3
    weight_decay: Annotated[float, Field(ge=0.0)] = 1.0e-4
    dropout_rate: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.5
    validation_strategy: Literal["lowest-train-subject"] = "lowest-train-subject"
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @model_validator(mode="after")
    def validate_torch_full(self) -> Self:
        if not self.architectures:
            raise ValueError("At least one Torch architecture is required")
        if len(set(self.architectures)) != len(self.architectures):
            raise ValueError("Torch architectures must be unique")
        if not self.spectral_methods:
            raise ValueError("At least one spectral method is required")
        if len(set(self.spectral_methods)) != len(self.spectral_methods):
            raise ValueError("Spectral methods must be unique")
        return self


class BNCIArtifactConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path = Path("artifacts/experiments/bnci2014_001")
    schema_version: Literal[1] = 1
    overwrite: bool = False


class BNCI2014001Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: BNCIDatasetConfig = BNCIDatasetConfig()
    split: BNCISplitConfig = BNCISplitConfig()
    baseline: BNCICSPBaselineConfig = BNCICSPBaselineConfig()
    project_features: BNCIProjectFeatureBenchmarkConfig = BNCIProjectFeatureBenchmarkConfig()
    torch_pilot: BNCITorchPilotConfig = BNCITorchPilotConfig()
    torch_full: BNCITorchFullBenchmarkConfig = BNCITorchFullBenchmarkConfig()
    artifacts: BNCIArtifactConfig = BNCIArtifactConfig()

    @model_validator(mode="after")
    def validate_artifact_policy(self) -> Self:
        if self.artifacts.overwrite:
            raise ValueError("BNCI2014_001 artifacts are immutable by default; overwrite is not supported")
        return self


def load_bnci_config(
    config_path: Path | None = None,
    *,
    overrides: dict[str, object] | None = None,
) -> BNCI2014001Config:
    resolved_path = config_path or Path("confs/experiments/bnci2014_001.yaml")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"BNCI2014_001 configuration does not exist: {resolved_path}")

    configs: list[object] = [OmegaConf.load(resolved_path)]
    if overrides:
        configs.append(OmegaConf.create(overrides))
    merged = OmegaConf.merge(*configs)
    payload = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved BNCI2014_001 configuration must be a mapping")
    return BNCI2014001Config.model_validate(payload)
