# Active Context

## Current Focus

Dataset API stages 0-5 are implemented:

- Strict FIF/label indexing by subject, trial, and block.
- Lazy MNE loading into typed NumPy samples.
- Versioned atomic disk cache and bounded in-process LRU cache.
- Explicit sequential or multiprocessing disk-cache warmup with structured reports.
- Executed tutorial notebooks `notebooks/1.0` through `notebooks/1.4`.
- Registered the standalone Data Analytics semantic layer
  `eeg-dataset-ml-experiments-semantic-layer` for dataset and experiment interpretation.

Spectral preprocessing checkpoints 1-2 are complete:

- `notebooks/2.0-dataset-overview.ipynb` is executed top-to-bottom.
- Full-corpus FIF metadata were audited for all 1,800 canonical blocks.
- Signal-level PSD, EOG quality, and topographies use a documented deterministic 16-block sample.
- Added strict OmegaConf/Pydantic configurations for FFT, Morlet, Superlet, and STFT.
- Added `PreprocessedDataset`, four method-specific dataset classes, `SpectralSample`, and validated
  transform-result contracts.
- Planned `notebooks/2.5-spectral-methods-comparison.ipynb` as the shared visual comparison of all
  four transforms after their implementations are complete.
- The current implementation plan is stored in
  `memory-bank/plans/2026-06-14-spectral-preprocessing.md`.
- Implementation is paused for user review before the spectral artifact cache is added.

## Next Actions

- Review checkpoint 2 and approve the spectral artifact cache stage.
- Implement atomic, source-aware caching for derived spectral samples.
- Define canonical train/validation/test split policy.
- Decide how labels from `labels.json` map to targets.
- Benchmark full-corpus cache warmup only when operational timing is needed.

## Open Questions

- What is the exact prediction target?
- Should evaluation be leave-one-subject-out, grouped K-fold by subject, within-subject, or multiple protocols?
- Which recordings are considered training, pattern/reference, validation, or test data?
