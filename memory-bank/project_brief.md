# Project Brief

## Purpose

This repository supports a master thesis project on EEG/EOG analysis with Python and machine learning.
The agent should help build a reproducible research codebase, not just isolated scripts.

## Current State

- `pyproject.toml` defines a Python >= 3.13 project with MNE, scientific Python, scikit-learn, and PyTorch.
- `utils/dataset.py` exists but is empty.
- `README.md` is empty.
- `data/` contains `.fif` EEG/EOG recordings organized by dataset split/type, subject, and trial.
- Some directories such as `confs/`, `notebooks/`, and `preprocessors/` exist but currently have no tracked files visible at shallow depth.

## Research Priorities

- Establish reliable data indexing and loading.
- Prevent train/test leakage across subjects, trials, and recording conditions.
- Make preprocessing choices explicit and reproducible.
- Build baselines before complex models.
- Keep metrics and experiment logs interpretable enough for thesis writing.
