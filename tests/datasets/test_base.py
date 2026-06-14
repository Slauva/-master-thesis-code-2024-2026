import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from utils.datasets.base import DatasetBase
from utils.datasets.schemas import GeometricSample, RandomSample


def _write_trial(
    dataset_dir: Path,
    *,
    subject_id: int,
    trial_number: int,
    prefix: str,
    blocks: list[dict],
    missing: tuple[str, int] | None = None,
) -> Path:
    trial_dir = dataset_dir / f"S_{subject_id}" / f"Trial_{trial_number}"
    trial_dir.mkdir(parents=True)
    (trial_dir / "labels.json").write_text(json.dumps({"blocks": blocks}), encoding="utf-8")

    for block in blocks:
        block_index = block["Exec_Block_Index"]
        for data_type in ("EEG", "EOG"):
            if missing == (data_type, block_index):
                continue
            (trial_dir / f"{prefix}_{data_type}_{block_index}.fif").touch()

    return trial_dir


@pytest.fixture
def label_blocks() -> list[dict]:
    return [
        {
            "Exec_Block_Index": 1,
            "type": "geometric",
            "pattern_id": 7,
            "img": [[0, 1], [1, 0]],
        },
        {
            "Exec_Block_Index": 2,
            "type": "random",
            "seed": 42,
            "img": [[1, 0], [0, 1]],
        },
    ]


def test_builds_nested_source_map_and_stable_flat_index(tmp_path: Path, label_blocks: list[dict]) -> None:
    _write_trial(
        tmp_path,
        subject_id=10,
        trial_number=2,
        prefix="exec",
        blocks=label_blocks,
    )
    _write_trial(
        tmp_path,
        subject_id=2,
        trial_number=1,
        prefix="exec",
        blocks=label_blocks[:1],
    )

    dataset = DatasetBase(tmp_path)

    assert len(dataset) == 3
    assert [(sample.subject_id, sample.trial_number, sample.block_index) for sample in dataset] == [
        (2, 1, 1),
        (10, 2, 1),
        (10, 2, 2),
    ]

    geometric = dataset.source_map[10][2][1]
    random = dataset[10, 2, 2]
    assert isinstance(geometric, GeometricSample)
    assert geometric.pattern_id == 7
    assert geometric.block_index == 1
    assert geometric.eeg_path.name == "exec_EEG_1.fif"
    assert geometric.eog_path.name == "exec_EOG_1.fif"
    assert isinstance(random, RandomSample)
    assert random.seed == 42


def test_uses_patt_filenames(tmp_path: Path, label_blocks: list[dict]) -> None:
    _write_trial(
        tmp_path,
        subject_id=1,
        trial_number=1,
        prefix="patt",
        blocks=label_blocks[:1],
    )

    sample = DatasetBase(tmp_path, dataset_step_type="patt")[1, 1, 1]

    assert sample.eeg_path.name == "patt_EEG_1.fif"
    assert sample.eog_path.name == "patt_EOG_1.fif"


def test_excludes_whole_subject_or_selected_trial(tmp_path: Path, label_blocks: list[dict]) -> None:
    for subject_id in (1, 2):
        for trial_number in (1, 2):
            _write_trial(
                tmp_path,
                subject_id=subject_id,
                trial_number=trial_number,
                prefix="exec",
                blocks=label_blocks[:1],
            )

    dataset = DatasetBase(
        tmp_path,
        exclude_samples={
            "S_1": [],
            "S_2": ["Trial_1"],
        },
    )

    assert len(dataset) == 1
    assert dataset[2, 2, 1].subject_id == 2
    assert 1 not in dataset.source_map
    assert 1 not in dataset.source_map[2]


@pytest.mark.parametrize(
    ("exclude_samples", "message"),
    [
        ({"subject-1": []}, "Invalid subject exclusion"),
        ({"S_1": ["trial-1"]}, "Invalid trial exclusion"),
    ],
)
def test_rejects_invalid_exclusions(
    tmp_path: Path,
    exclude_samples: dict[str, list[str]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DatasetBase(tmp_path, exclude_samples=exclude_samples)


def test_rejects_missing_label_file(tmp_path: Path) -> None:
    (tmp_path / "S_1" / "Trial_1").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Label file does not exist"):
        DatasetBase(tmp_path)


def test_rejects_missing_fif_file(tmp_path: Path, label_blocks: list[dict]) -> None:
    _write_trial(
        tmp_path,
        subject_id=1,
        trial_number=1,
        prefix="exec",
        blocks=label_blocks[:1],
        missing=("EOG", 1),
    )

    with pytest.raises(FileNotFoundError, match="exec_EOG_1.fif"):
        DatasetBase(tmp_path)


def test_rejects_duplicate_block_index(tmp_path: Path, label_blocks: list[dict]) -> None:
    duplicate_blocks = [label_blocks[0], {**label_blocks[1], "Exec_Block_Index": 1}]
    _write_trial(
        tmp_path,
        subject_id=1,
        trial_number=1,
        prefix="exec",
        blocks=duplicate_blocks,
    )

    with pytest.raises(ValueError, match="Duplicate block index 1"):
        DatasetBase(tmp_path)


def test_rejects_unknown_sample_type(tmp_path: Path, label_blocks: list[dict]) -> None:
    invalid_block = {**label_blocks[0], "type": "unknown"}
    _write_trial(
        tmp_path,
        subject_id=1,
        trial_number=1,
        prefix="exec",
        blocks=[invalid_block],
    )

    with pytest.raises(ValidationError):
        DatasetBase(tmp_path)


def test_tuple_lookup_reports_missing_sample(tmp_path: Path, label_blocks: list[dict]) -> None:
    _write_trial(
        tmp_path,
        subject_id=1,
        trial_number=1,
        prefix="exec",
        blocks=label_blocks[:1],
    )
    dataset = DatasetBase(tmp_path)

    with pytest.raises(KeyError, match="subject=1, trial=1, block=99"):
        dataset[1, 1, 99]

    with pytest.raises(TypeError, match="must be a"):
        dataset[1]  # type: ignore[index]


@pytest.mark.parametrize(
    ("dataset_dir", "dataset_step_type", "expected"),
    [
        ("Data_Train", "exec", 1260),
        ("Data_Pattern", "patt", 540),
    ],
)
def test_indexes_complete_real_dataset(dataset_dir: str, dataset_step_type: str, expected: int) -> None:
    root = Path(__file__).resolve().parents[2] / "data" / dataset_dir
    if not root.exists():
        pytest.skip(f"Real dataset is unavailable: {root}")

    dataset = DatasetBase(root, dataset_step_type=dataset_step_type)  # type: ignore[arg-type]

    assert len(dataset) == expected
