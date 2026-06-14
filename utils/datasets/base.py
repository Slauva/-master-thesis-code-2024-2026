import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Literal

from utils.datasets.proto import DatasetProto
from utils.datasets.schemas import LabelModel, Sample

SampleKey = tuple[int, int, int]
SourceMap = dict[int, dict[int, dict[int, Sample]]]


class DatasetBase(DatasetProto):
    def __init__(
        self,
        dataset_dir: Path,
        *,
        dataset_step_type: Literal["exec", "patt"] = "exec",
        exclude_samples: dict[str, list[str]] | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_step_type = dataset_step_type
        self.exclude_samples = self._validate_exclusions(exclude_samples or {})

        if not self.dataset_dir.is_dir():
            raise NotADirectoryError(f"Dataset directory does not exist: {self.dataset_dir}")

        source_map, samples = self._build_index()
        self.source_map = source_map
        self.samples = samples
        self._sample_by_key = MappingProxyType(
            {(sample.subject_id, sample.trial_number, sample.block_index): sample for sample in samples}
        )

    def __getitem__(self, key: SampleKey) -> Sample:
        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError("DatasetBase key must be a (subject_id, trial_number, block_index) tuple")

        try:
            return self._sample_by_key[key]
        except KeyError as error:
            raise KeyError(f"Sample does not exist: subject={key[0]}, trial={key[1]}, block={key[2]}") from error

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _build_index(self) -> tuple[SourceMap, tuple[Sample, ...]]:
        source_map: SourceMap = {}
        samples: list[Sample] = []

        trial_paths = sorted(
            self.dataset_dir.glob(f"{self.GLOB_SUBJECT_FOLDER_PATTERN}/{self.GLOB_TRIAL_FOLDER_PATTERN}"),
            key=self._extract_keys,
        )
        for trial_path in trial_paths:
            subject_id, trial_number = self._extract_keys(trial_path)
            if self._is_excluded(trial_path):
                continue

            trial_samples = self._read_label_file(
                trial_path / self.LABEL_FILE_PATTERN,
                subject_id=subject_id,
                trial_number=trial_number,
            )
            trial_map: dict[int, Sample] = {}
            for sample in trial_samples:
                if sample.block_index in trial_map:
                    raise ValueError(
                        f"Duplicate block index {sample.block_index} in {trial_path / self.LABEL_FILE_PATTERN}"
                    )
                self._validate_sample_files(sample)
                trial_map[sample.block_index] = sample
                samples.append(sample)

            source_map.setdefault(subject_id, {})[trial_number] = trial_map

        return source_map, tuple(samples)

    def _read_label_file(self, filepath: Path, *, subject_id: int, trial_number: int) -> list[Sample]:
        if not filepath.is_file():
            raise FileNotFoundError(f"Label file does not exist: {filepath}")

        with filepath.open(encoding="utf-8") as file:
            raw = json.load(file)

        blocks = raw.get("blocks") if isinstance(raw, dict) else None
        if not isinstance(blocks, list):
            raise ValueError(f"Label file must contain a 'blocks' list: {filepath}")

        enriched_blocks = []
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError(f"Every label block must be an object: {filepath}")

            block_index = block.get("Exec_Block_Index")
            if not isinstance(block_index, int) or isinstance(block_index, bool):
                raise ValueError(f"Invalid Exec_Block_Index in {filepath}: {block_index!r}")

            enriched_blocks.append(
                {
                    **block,
                    "subject_id": subject_id,
                    "trial_number": trial_number,
                    "eeg_path": filepath.parent / self._get_exg_filename(block_index, data_type="EEG"),
                    "eog_path": filepath.parent / self._get_exg_filename(block_index, data_type="EOG"),
                }
            )

        return LabelModel.model_validate({"blocks": enriched_blocks}).blocks

    def _get_exg_filename(self, index: int, *, data_type: Literal["EEG", "EOG"] = "EEG") -> str:
        return self.FIF_FILE_PATTERN % (self.dataset_step_type, data_type, index)

    def _is_excluded(self, trial_path: Path) -> bool:
        excluded_trials = self.exclude_samples.get(trial_path.parent.name)
        return excluded_trials is not None and (not excluded_trials or trial_path.name in excluded_trials)

    def _validate_exclusions(self, exclude_samples: dict[str, list[str]]) -> dict[str, frozenset[str]]:
        validated: dict[str, frozenset[str]] = {}
        for subject, trials in exclude_samples.items():
            if re.fullmatch(self.RE_SUBJECT_FOLDER_PATTERN, subject) is None:
                raise ValueError(f"Invalid subject exclusion {subject!r}; expected format 'S_<number>'")

            for trial in trials:
                if re.fullmatch(self.RE_TRIAL_FOLDER_PATTERN, trial) is None:
                    raise ValueError(f"Invalid trial exclusion {trial!r}; expected format 'Trial_<number>'")
            validated[subject] = frozenset(trials)

        return validated

    @staticmethod
    def _validate_sample_files(sample: Sample) -> None:
        missing = [path for path in (sample.eeg_path, sample.eog_path) if not path.is_file()]
        if missing:
            missing_paths = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Missing FIF file(s) for subject={sample.subject_id}, trial={sample.trial_number}, "
                f"block={sample.block_index}: {missing_paths}"
            )

    def _extract_keys(self, path: Path) -> tuple[int, int]:
        subject_name = path.parent.name
        trial_name = path.name

        if re.fullmatch(self.RE_SUBJECT_FOLDER_PATTERN, subject_name) is None:
            raise ValueError(f"Could not parse subject folder: {path}")
        if re.fullmatch(self.RE_TRIAL_FOLDER_PATTERN, trial_name) is None:
            raise ValueError(f"Could not parse trial folder: {path}")

        return int(subject_name.split("_", maxsplit=1)[1]), int(trial_name.split("_", maxsplit=1)[1])
