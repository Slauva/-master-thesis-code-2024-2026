---
name: eeg-ml-research
description: Use for EEG/EOG analysis, MNE workflows, dataset loading, preprocessing, feature extraction, model training, validation design, leakage checks, and thesis-grade ML experiments in this repository.
---

# EEG ML Research

## Default Stance

Be conservative: EEG results are easy to overstate and easy to contaminate with leakage.
Prefer a clear reproducible baseline over a complex model with unclear splits.

## Workflow

1. Inspect data organization and label semantics before writing modeling code.
2. Define the prediction unit: sample, window, trial, session, or subject.
3. Define leakage boundaries before splitting data.
4. Build deterministic data indexing and loading with `pathlib.Path` and MNE APIs.
5. Make preprocessing explicit: channel selection, filters, resampling, artifact handling, epochs/windows, normalization, and rejection.
6. Start with simple baselines before neural models.
7. Report metrics at the same level as the claim: window, trial, subject, or cohort.

## Leakage Checklist

- Do not split windows from the same trial across train and validation unless the claim is explicitly within-trial.
- Do not let the same subject appear in train and test for subject-generalization claims.
- Fit scalers, PCA, feature selectors, artifact models, and thresholds on train folds only.
- Keep label-derived operations out of preprocessing unless they are inside the training fold.
- Track subject, trial, condition, and source path in metadata.

## Recommended Evaluation

- For subject generalization: `GroupKFold`, `LeaveOneGroupOut`, or fixed held-out subjects using subject id as the group.
- For limited data: report fold-level results and mean/std; avoid a single lucky split.
- For class imbalance: report balanced accuracy, macro F1, per-class recall, and confusion matrix.
- For probabilistic models: include calibration or threshold-independent metrics when useful.

## Implementation Preferences

- Keep dataset indexing separate from signal loading.
- Use lazy loading where full FIF arrays would be large.
- Return structured records with paths, subject, trial, modality, run id, and label metadata.
- Store configs in typed structures; avoid implicit constants scattered through notebooks.
- Keep notebooks exploratory and move reusable code into modules.
