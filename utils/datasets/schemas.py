from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class RawSample(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    subject_id: int = Field(ge=1)
    trial_number: int = Field(ge=1)
    block_index: int = Field(alias="Exec_Block_Index", ge=1)
    eeg_path: Path
    eog_path: Path
    img: list[list[int]]


class GeometricSample(RawSample):
    type: Literal["geometric"] = "geometric"
    pattern_id: int


class RandomSample(RawSample):
    type: Literal["random"] = "random"
    seed: int


Sample = Annotated[Union[GeometricSample, RandomSample], Field(discriminator="type")]


class LabelModel(BaseModel):
    blocks: list[Sample]


@dataclass(frozen=True, slots=True)
class LoadedSample:
    sample: Sample
    eeg: NDArray[np.floating[Any]]
    eog: NDArray[np.floating[Any]]
    sfreq: float
    eeg_channels: tuple[str, ...]
    eog_channels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CacheWarmupError:
    key: tuple[int, int, int]
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class CacheWarmupReport:
    processed: int
    cached: int
    skipped: int
    failed: int
    errors: tuple[CacheWarmupError, ...]
    max_workers: int
    duration_seconds: float

    @property
    def total(self) -> int:
        return self.processed + self.cached + self.skipped + self.failed
