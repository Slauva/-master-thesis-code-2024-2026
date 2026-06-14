from pathlib import Path
from typing import Literal

from utils.datasets.base import DatasetBase


class NumpyDataset(DatasetBase):
    def __init__(
        self,
        dataset_dir: Path,
        *,
        dataset_step_type: Literal["exec", "patt"] = "exec",
        dataset_pattern_type: Literal["geometric", "random"] | None = None,
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

    def _get_paths(self) -> list[str]:
        pass
