from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import mne
import numpy as np
from numpy.typing import DTypeLike

from utils.datasets.base import DatasetBase, SampleKey
from utils.datasets.schemas import LoadedSample, Sample


class NumpyDataset(DatasetBase):
    def __init__(
        self,
        dataset_dir: Path,
        *,
        dataset_step_type: Literal["exec", "patt"] = "exec",
        dataset_pattern_type: Literal["geometric", "random"] | None = None,
        dtype: DTypeLike = np.float32,
        cache_policy: Literal["memory", "disk", "both"] | None = None,
        cache_dir: Path | None = None,
        preload: bool = False,
        exclude_samples: dict[str, list[str]] | None = None,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            dataset_step_type=dataset_step_type,
            exclude_samples=exclude_samples,
        )

        # Caching variables
        if cache_policy in ["disk", "both"] and cache_dir is None:
            raise ValueError("`cache_dir` value cannot be None if `cache_policy` value is `disk` or `both`")

        self.cache_policy = cache_policy
        self.cache_dir = cache_dir
        self.preload = preload

        # Filtration variables
        self.dataset_pattern_type = dataset_pattern_type
        if dataset_pattern_type is not None:
            filtered_samples = tuple(sample for sample in self.samples if sample.type == dataset_pattern_type)
            self._filter_index(filtered_samples)

        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f"`dtype` must be float32 or float64, got {self.dtype}")

    def __getitem__(self, key: int | SampleKey) -> LoadedSample:
        sample = self._get_sample(key)
        return self._load_sample(sample)

    def __iter__(self) -> Iterator[LoadedSample]:
        for index in range(len(self)):
            yield self[index]

    def _get_sample(self, key: int | SampleKey) -> Sample:
        if isinstance(key, bool):
            raise TypeError("NumpyDataset key must be an integer or a (subject_id, trial_number, block_index) tuple")
        if isinstance(key, int):
            return self.samples[key]
        return super().__getitem__(key)

    def _load_sample(self, sample: Sample) -> LoadedSample:
        eeg_raw = mne.io.read_raw_fif(sample.eeg_path, preload=False, verbose="ERROR")
        eog_raw = mne.io.read_raw_fif(sample.eog_path, preload=False, verbose="ERROR")
        try:
            self._validate_raw_pair(sample, eeg_raw, eog_raw)
            eeg = eeg_raw.get_data().astype(self.dtype, copy=False)
            eog = eog_raw.get_data().astype(self.dtype, copy=False)
            return LoadedSample(
                sample=sample,
                eeg=eeg,
                eog=eog,
                sfreq=float(eeg_raw.info["sfreq"]),
                eeg_channels=tuple(eeg_raw.ch_names),
                eog_channels=tuple(eog_raw.ch_names),
            )
        finally:
            eeg_raw.close()
            eog_raw.close()

    @staticmethod
    def _validate_raw_pair(sample: Sample, eeg_raw: mne.io.BaseRaw, eog_raw: mne.io.BaseRaw) -> None:
        eeg_sfreq = float(eeg_raw.info["sfreq"])
        eog_sfreq = float(eog_raw.info["sfreq"])
        if not np.isclose(eeg_sfreq, eog_sfreq):
            raise ValueError(
                f"EEG/EOG sampling frequencies differ for subject={sample.subject_id}, "
                f"trial={sample.trial_number}, block={sample.block_index}: {eeg_sfreq} != {eog_sfreq}"
            )
        if eeg_raw.n_times != eog_raw.n_times:
            raise ValueError(
                f"EEG/EOG sample counts differ for subject={sample.subject_id}, "
                f"trial={sample.trial_number}, block={sample.block_index}: "
                f"{eeg_raw.n_times} != {eog_raw.n_times}"
            )
