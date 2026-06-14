from typing import Protocol

from utils.datasets.schemas import Sample


class DatasetProto(Protocol):
    SUBJECT_FOLDER_PATTERN = "S_%d"
    RE_SUBJECT_FOLDER_PATTERN = "S_\\d+"
    GLOB_SUBJECT_FOLDER_PATTERN = "S_*"

    TRIAL_FOLDER_PATTERN = "Trial_%d"
    RE_TRIAL_FOLDER_PATTERN = "Trial_\\d+"
    GLOB_TRIAL_FOLDER_PATTERN = "Trial_*"

    FIF_FILE_PATTERN = "%s_%s_%d.fif"
    LABEL_FILE_PATTERN = "labels.json"

    def __getitem__(self, key: tuple[int, int, int]) -> Sample: ...

    def __len__(self) -> int: ...
