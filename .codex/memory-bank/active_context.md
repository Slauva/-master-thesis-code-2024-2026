# Active Context

## Current Focus

Dataset API stages 0-6 are implemented:

- Strict FIF/label indexing by subject, trial, and block.
- Lazy MNE loading into typed NumPy samples.
- Versioned atomic disk cache and bounded in-process LRU cache.
- Explicit sequential or multiprocessing disk-cache warmup with structured reports.
- Executed tutorial notebooks `notebooks/1.0` through `notebooks/1.4`.
- Registered the standalone Data Analytics semantic layer
  `eeg-dataset-ml-experiments-semantic-layer` for dataset and experiment interpretation.

Spectral preprocessing checkpoints 1-9 are complete:

- `notebooks/2.0-dataset-overview.ipynb` is executed top-to-bottom.
- Full-corpus FIF metadata were audited for all 1,800 canonical blocks.
- Signal-level PSD, EOG quality, and topographies use a documented deterministic 16-block sample.
- Added strict OmegaConf/Pydantic configurations for FFT, Morlet, Superlet, and STFT.
- Added `PreprocessedDataset`, four method-specific dataset classes, `SpectralSample`, and validated
  transform-result contracts.
- Added atomic disk caching for spectral power, frequency axes, optional time axes, and manifests;
  original EOG continues to come from `NumpyDataset` and is not duplicated.
- Spectral cache identity covers resolved config, source dtype, schema, transform class, and
  transform version; manifests additionally validate both source FIF signatures.
- Implemented FFT preprocessing with polyphase resampling to 125 Hz, channel-wise demeaning, a
  periodic Hann window, one-sided density scaling, and power-preserving overlap rebinning onto the
  exact 2-40 Hz, 1 Hz grid.
- Added and executed `notebooks/2.1-fft.ipynb` with synthetic 10 Hz and 23 Hz validation plus one
  canonical `exec` block and one canonical `patt` block.
- FFT outputs have shape `(channel, frequency)`, use `float32`, have no time axis, and occupy about
  12 KiB per demonstrated 63-channel block including the manifest.
- Implemented Morlet wavelet power with `n_cycles=clip(frequency / 2, 3, 10)`, zero-mean wavelets,
  FFT convolution, a common 149-sample edge trim per side, and centered 32-sample power bins.
- Added and executed `notebooks/2.2-morlet.ipynb`; synthetic 10 Hz and 25 Hz bursts were localized
  at the correct frequencies with peak-time center errors below 0.06 s.
- Real Morlet outputs have shapes `(63, 39, 53)` for the demonstrated `exec` block and
  `(63, 39, 92)` for the demonstrated `patt` block, with a 0.256 s time step.
- Demonstrated Morlet cache entries occupy about 512 KiB for `exec` and 886 KiB for `patt`.
- Reworked the existing Superlet implementation into a typed fractional adaptive transform pinned
  to the documented upstream revision and accompanied by its MIT license notice.
- Default Superlet uses adaptive order 1-10 with `c_1=3`, stores coefficient magnitude squared,
  trims 199 samples per side from the longest contributing fractional-order support, and averages
  centered 32-sample power bins.
- Added and executed `notebooks/2.3-superlet.ipynb`; stationary 20 Hz and 24 Hz tones were resolved
  as separate local maxima with a lower 22 Hz midpoint.
- Real Superlet outputs have shapes `(63, 39, 50)` for the demonstrated `exec` block and
  `(63, 39, 89)` for the demonstrated `patt` block, with a 0.256 s time step.
- Demonstrated Superlet cache entries occupy about 483 KiB for `exec` and 857 KiB for `patt`.
- Implemented STFT PSD with a periodic 2 s Hann window, 32-sample hop, `mfft=250`,
  `fft_mode="onesided2X"`, and exclusion of every slice affected by border padding.
- Shared power-preserving density rebinning between FFT and STFT; STFT native 0.5 Hz bins are
  rebinned onto the exact 2-40 Hz, 1 Hz output grid instead of being subsampled.
- Added and executed `notebooks/2.4-stft.ipynb`; synthetic 10 Hz and 25 Hz bursts were localized
  at the correct frequencies and intervals, while integrated stationary-sine PSD recovered the
  expected mean square.
- Real STFT outputs have shapes `(63, 39, 55)` for the demonstrated `exec` block and
  `(63, 39, 94)` for the demonstrated `patt` block, with a 0.256 s time step.
- Demonstrated STFT cache entries occupy about 531 KiB for `exec` and 905 KiB for `patt`.
- Added and executed `notebooks/2.5-spectral-methods-comparison.ipynb` on one shared synthetic
  signal and canonical key `(1, 1, 1)` from both recording families.
- The comparison validates global FFT peaks at 10 Hz and 25 Hz plus correct burst-frequency and
  burst-interval localization for Morlet, Superlet, and STFT.
- Real-data panels use the same `Fp1` channel and 2-14 s interval across methods. Native
  method-specific marginals remain in PSD or wavelet-power units; shared maps and time marginals
  use explicitly display-only per-method normalization.
- The comparison reports output shape, axis resolution, direct single-channel runtime, and
  full 63-channel cache size without treating runtime as a scientific quality ranking.
- Added method-specific dimensionality enforcement at the dataset boundary: FFT is strictly 2D;
  Morlet, Superlet, and STFT are strictly 3D.
- Added automated integration checks that notebooks `2.1` through `2.5` are executed, contain no
  error outputs, and include both `exec` and `patt` canonical demonstrations.
- Revalidated all eight canonical cached outputs through the public dataset classes. Shapes,
  frequency grids, dtypes, scaling contracts, source keys, and non-duplication of EOG all passed.
- Estimated logical full-corpus storage for 1,800 blocks as 21.10 MiB FFT, 1.070 GiB Morlet,
  1.020 GiB Superlet, and 1.103 GiB STFT; all four methods total about 3.214 GiB (3.451 GB).
- The current implementation plan is stored in
  `.codex/memory-bank/plans/2026-06-14-spectral-preprocessing.md`.
- Spectral preprocessing plan checkpoints 1-9 are complete.

PyTorch dataset checkpoints 1-4 are complete:

- Added immutable tensor sample schemas for raw and spectral data.
- Added raw and spectral batch schemas with tensor-only `.pin_memory()` and `.to()` operations.
- Kept source metadata, channel names, method, scaling, and sampling-rate metadata outside device
  transfer.
- Reserved independent spectral and original-EOG length/mask fields because real blocks have
  variable durations and spectral transforms produce method-specific time axes.
- Added `TorchDataset` as a zero-copy map-style adapter over `NumpyDataset`.
- Added strict raw collation with zero padding, lengths, valid-time masks, and EOG finite masks.
- Preserved integer and canonical tuple indexing plus source `samples` and `source_map`.
- Added and executed `notebooks/3.0-torch-dataset-gpu.ipynb` with a mixed canonical
  `Data_Train/exec` and `Data_Pattern/patt` batch.
- Verified pinned host memory, non-blocking CUDA transfer, and a finite `Conv1d` forward/backward
  pass on the available CUDA device.
- Added `TorchPreprocessedDataset` as a zero-copy adapter over all four spectral dataset classes.
- FFT batches stack without a time axis; Morlet, Superlet, and STFT use padded per-sample time
  coordinates, spectral lengths, and spectral time masks.
- Original EOG uses independent padding, lengths, time masks, and finite-value masks.
- Added and executed `notebooks/3.1-torch-preprocessed-dataset-gpu.ipynb` for all four methods and
  both canonical recording families.
- Verified pinned host memory, non-blocking CUDA transfer, and finite `Conv2d` forward/backward
  passes for FFT, Morlet, Superlet, and STFT.
- Added 33 focused PyTorch tests across raw and spectral adapters; Ruff passes and the full suite
  reports 150 passed before integration.
- Added automated integration checks for both PyTorch notebooks: all code cells must be executed,
  stored outputs must be error-free and contain `CUDA_VERIFIED`, and both canonical recording
  families must be present.
- Revalidated the complete public API and repository. Ruff passes and the full suite reports
  152 passed; two existing Python 3.13 multiprocessing `fork()` deprecation warnings remain in
  disk-cache tests.
- Stored the implementation plan in
  `.codex/memory-bank/plans/2026-06-14-torch-datasets-gpu.md`.
- PyTorch dataset plan checkpoints 1-4 are complete.

Project planning workflow:

- Added the project-local `manage-staged-plans` skill.
- New substantial plans are split into reviewable stages and saved under
  `.codex/memory-bank/plans/` only after explicit user approval.
- Every implemented stage stops in `Awaiting Review`; it becomes `Completed` only after explicit
  user approval, with progress and relevant decisions or experiments written back to memory.
- Quantitative or scientific results that benefit from visualization use the
  `data-analytics:jupyter-notebooks` workflow and an executed notebook under `notebooks/`.

## Next Actions

- Define canonical train/validation/test split policy.
- Decide how labels from `labels.json` map to targets.
- Define whether training examples are whole blocks or fixed windows after the split policy is set.
- Benchmark full-corpus cache warmup only when operational timing is needed.

## Open Questions

- What is the exact prediction target?
- Should evaluation be leave-one-subject-out, grouped K-fold by subject, within-subject, or multiple protocols?
- Which recordings are considered training, pattern/reference, validation, or test data?
