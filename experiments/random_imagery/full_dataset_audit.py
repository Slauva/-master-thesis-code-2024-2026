import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from experiments.logistic_regression import (
    EvaluationDirection,
    EvaluationProtocolDefinition,
    SubjectSplit,
    audit_evaluation_direction,
    audit_subject_split,
    build_random_imagery_targets,
)
from experiments.random_imagery.config import DatasetSelectionConfig, SubjectSplitConfig
from utils.datasets import NumpyDataset
from utils.datasets.schemas import GeometricSample, RandomSample, Sample


def build_full_dataset_audit(
    *,
    dataset_config: DatasetSelectionConfig | None = None,
    split_config: SubjectSplitConfig | None = None,
) -> dict[str, Any]:
    dataset_config = dataset_config or DatasetSelectionConfig(pattern_type=None)
    split_config = split_config or SubjectSplitConfig()
    if dataset_config.pattern_type is not None:
        raise ValueError("Full imagery audit requires dataset_config.pattern_type=None")

    dataset = NumpyDataset(
        dataset_config.dataset_dir,
        dataset_step_type=dataset_config.recording_family,
        dataset_pattern_type=dataset_config.pattern_type,
        cache_policy="none",
    )
    targets = build_random_imagery_targets(
        dataset.samples,
        image_rows=dataset_config.image_rows,
        image_columns=dataset_config.image_columns,
        allowed_sample_types=dataset_config.target_sample_types,
    )
    cross_subject = _build_cross_subject_definition(
        targets,
        split_config=split_config,
    )
    within_subject = _build_within_subject_definition(
        targets,
    )
    has_blocking_leakage = any(
        audit.has_forbidden_leakage
        for protocol in (cross_subject, within_subject)
        for audit in protocol.audits
    )
    return {
        "stage1_status": "blocked" if has_blocking_leakage else "ready",
        "dataset": _dataset_summary(dataset.samples),
        "targets": {
            "n_rows": int(targets.y.shape[0]),
            "n_pixels": int(targets.y.shape[1]),
            "pixel_names": list(targets.pixel_names),
            "positive_counts": _support_summary(targets.y),
            "random_seed_rows": int(np.count_nonzero(targets.seeds >= 0)),
            "missing_random_seed_rows": int(np.count_nonzero(targets.seeds < 0)),
        },
        "protocols": {
            "cross_subject": _protocol_summary(cross_subject),
            "within_subject": _protocol_summary(within_subject),
        },
    }


def _build_cross_subject_definition(
    targets: Any,
    *,
    split_config: SubjectSplitConfig,
) -> EvaluationProtocolDefinition:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=split_config.test_size,
        random_state=split_config.random_state,
    )
    rows = np.arange(targets.y.shape[0], dtype=np.int64)
    train_indices, test_indices = next(
        splitter.split(rows[:, np.newaxis], groups=targets.subject_ids)
    )
    train_indices = np.sort(train_indices.astype(np.int64, copy=False))
    test_indices = np.sort(test_indices.astype(np.int64, copy=False))
    split = SubjectSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        train_subjects=tuple(int(value) for value in np.unique(targets.subject_ids[train_indices])),
        test_subjects=tuple(int(value) for value in np.unique(targets.subject_ids[test_indices])),
        n_samples=targets.y.shape[0],
        random_state=split_config.random_state,
        test_size=split_config.test_size,
    )
    legacy_audit = audit_subject_split(targets, split)
    eligible_subjects = tuple(int(value) for value in np.unique(targets.subject_ids))
    direction = EvaluationDirection(
        protocol="cross-subject",
        name="cross-subject",
        label="cross-subject",
        train_indices=split.train_indices,
        test_indices=split.test_indices,
        train_subjects=split.train_subjects,
        test_subjects=split.test_subjects,
        eligible_subjects=eligible_subjects,
        excluded_subjects=(),
        n_samples=targets.y.shape[0],
    )
    audit = audit_evaluation_direction(targets, direction)
    if tuple(legacy_audit.overlapping_image_fingerprints) != tuple(audit.overlapping_image_fingerprints):
        raise RuntimeError("Cross-subject audit implementations disagree on image fingerprints")
    return EvaluationProtocolDefinition(
        protocol="cross-subject",
        label="cross-subject",
        eligible_subjects=eligible_subjects,
        excluded_subjects=(),
        directions=(direction,),
        audits=(audit,),
    )


def _build_within_subject_definition(targets: Any) -> EvaluationProtocolDefinition:
    all_subjects = tuple(int(value) for value in np.unique(targets.subject_ids))
    eligible_subjects = tuple(
        subject
        for subject in all_subjects
        if {1, 2}
        <= set(
            int(value)
            for value in np.unique(
                targets.trial_numbers[targets.subject_ids == subject]
            )
        )
    )
    excluded_subjects = tuple(
        subject for subject in all_subjects if subject not in set(eligible_subjects)
    )
    eligible_mask = np.isin(targets.subject_ids, eligible_subjects)
    directions = []
    audits = []
    for name, train_trial, test_trial in (
        ("trial-1-to-trial-2", 1, 2),
        ("trial-2-to-trial-1", 2, 1),
    ):
        train_indices = np.flatnonzero(
            eligible_mask & (targets.trial_numbers == train_trial)
        ).astype(np.int64, copy=False)
        test_indices = np.flatnonzero(
            eligible_mask & (targets.trial_numbers == test_trial)
        ).astype(np.int64, copy=False)
        train_indices.setflags(write=False)
        test_indices.setflags(write=False)
        direction = EvaluationDirection(
            protocol="within-subject",
            name=name,
            label=f"Trial {train_trial} -> Trial {test_trial}",
            train_indices=train_indices,
            test_indices=test_indices,
            train_subjects=eligible_subjects,
            test_subjects=eligible_subjects,
            eligible_subjects=eligible_subjects,
            excluded_subjects=excluded_subjects,
            n_samples=targets.y.shape[0],
            train_trial=train_trial,
            test_trial=test_trial,
        )
        directions.append(direction)
        audits.append(audit_evaluation_direction(targets, direction))

    return EvaluationProtocolDefinition(
        protocol="within-subject",
        label="identity-overlapping bidirectional cross-trial",
        eligible_subjects=eligible_subjects,
        excluded_subjects=excluded_subjects,
        directions=tuple(directions),
        audits=tuple(audits),
    )


def write_full_dataset_audit(
    output_path: Path,
    *,
    dataset_config: DatasetSelectionConfig | None = None,
    split_config: SubjectSplitConfig | None = None,
) -> dict[str, Any]:
    audit = build_full_dataset_audit(
        dataset_config=dataset_config,
        split_config=split_config,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _dataset_summary(samples: Sequence[Sample]) -> dict[str, Any]:
    sample_type_counts = Counter(sample.type for sample in samples)
    subjects = sorted({sample.subject_id for sample in samples})
    trials = sorted({(sample.subject_id, sample.trial_number) for sample in samples})
    return {
        "n_rows": len(samples),
        "sample_type_counts": dict(sorted(sample_type_counts.items())),
        "n_subjects": len(subjects),
        "subject_ids": subjects,
        "n_subject_trials": len(trials),
        "n_random_seed_values": len({sample.seed for sample in samples if isinstance(sample, RandomSample)}),
        "n_geometric_pattern_ids": len(
            {sample.pattern_id for sample in samples if isinstance(sample, GeometricSample)}
        ),
    }


def _support_summary(y: np.ndarray) -> dict[str, int]:
    values = np.asarray(y)
    positive_counts = values if values.ndim == 1 else values.sum(axis=0, dtype=np.int64)
    return {
        "min": int(positive_counts.min()),
        "max": int(positive_counts.max()),
        "total": int(positive_counts.sum()),
    }


def _protocol_summary(protocol: Any) -> dict[str, Any]:
    return {
        "label": protocol.label,
        "eligible_subjects": list(protocol.eligible_subjects),
        "excluded_subjects": list(protocol.excluded_subjects),
        "directions": [
            {
                "name": direction.name,
                "n_train_rows": int(direction.train_indices.size),
                "n_test_rows": int(direction.test_indices.size),
                "train_subjects": list(direction.train_subjects),
                "test_subjects": list(direction.test_subjects),
                "train_trial": direction.train_trial,
                "test_trial": direction.test_trial,
                "audit": {
                    "has_forbidden_leakage": audit.has_forbidden_leakage,
                    "all_tasks_have_both_classes": audit.all_tasks_have_both_classes,
                    "overlapping_subjects": list(audit.overlapping_subjects),
                    "overlapping_sample_keys": [list(key) for key in audit.overlapping_sample_keys],
                    "overlapping_seeds": list(audit.overlapping_seeds),
                    "overlapping_image_fingerprints": list(audit.overlapping_image_fingerprints),
                    "overlapping_random_image_fingerprints": list(
                        audit.overlapping_random_image_fingerprints
                    ),
                    "overlapping_geometric_pattern_ids": list(audit.overlapping_geometric_pattern_ids),
                    "overlapping_trial_numbers": list(audit.overlapping_trial_numbers),
                    "train_positive_counts": _support_summary(audit.train_positive_counts),
                    "test_positive_counts": _support_summary(audit.test_positive_counts),
                    "subject_contract_satisfied": audit.subject_contract_satisfied,
                    "trial_contract_satisfied": audit.trial_contract_satisfied,
                },
            }
            for direction, audit in zip(protocol.directions, protocol.audits, strict=True)
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="full-imagery-audit",
        description="Audit full geometric+random imagery targets and leakage-aware protocols.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/full-imagery/stage1_full_dataset_audit.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_full_dataset_audit(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
