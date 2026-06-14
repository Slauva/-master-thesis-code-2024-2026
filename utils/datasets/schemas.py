from pathlib import Path
from typing import Annotated, Literal, Union

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
