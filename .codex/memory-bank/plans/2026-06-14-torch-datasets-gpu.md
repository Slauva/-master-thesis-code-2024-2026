# PyTorch Datasets And GPU Verification Plan

Status: in progress
Last updated: 2026-06-14
Next checkpoint: 3 - preprocessed PyTorch dataset

## Public Contract

- `TorchDataset(NumpyDataset(...))` returns `TorchSample` with EEG/EOG tensors and source metadata.
- `TorchPreprocessedDataset(FFTDataset/MorletDataset/SuperletDataset/STFTDataset)` returns
  `TorchSpectralSample`.
- NumPy arrays are converted with `torch.from_numpy`; source dtype is preserved and CUDA work is
  kept outside dataset workers.
- `collate_torch_samples()` pads raw EEG/EOG time axes and returns lengths, a valid-time mask, and
  an EOG finite-value mask.
- `collate_torch_spectral_samples()` stacks FFT output and pads the time axis for Morlet, Superlet,
  and STFT. Spectral and original-EOG lengths and masks remain independent.
- Typed batches expose `.pin_memory()` and `.to(device, non_blocking=True)`. Signal tensors and
  tensor axes move to the selected device; source metadata and channel names remain CPU objects.
- Collators reject incompatible methods, channels, sampling rates, frequency grids, or scaling.
- No ML target or train/validation/test split is inferred in this phase.

## Checkpoints

### 1. Shared Contract - Completed

- Added immutable `TorchSample`, `TorchSpectralSample`, `TorchSampleBatch`, and
  `TorchSpectralBatch` schemas.
- Defined raw padding metadata and independent spectral/EOG padding metadata.
- Added batch `.pin_memory()` and `.to()` contracts without moving source metadata.
- Exported the schemas from `utils.datasets`.
- Saved this plan and updated the active project context.
- Stop for user review before checkpoint 2.

### 2. Raw PyTorch Dataset - Completed

- Implemented `TorchDataset` as a map-style adapter over an already configured `NumpyDataset`.
- Tensor conversion uses `torch.from_numpy`, preserves `float32`/`float64`, and shares source
  array storage without adding another cache.
- Implemented raw collation with zero padding, lengths, valid-time masks, and EOG finite masks.
- Added strict validation for tensor shape, finite EEG, channel order, dtype, sampling frequency,
  and CPU-only pre-collation storage.
- Preserved integer and canonical tuple indexing plus `samples` and `source_map` proxies.
- Added 12 focused tests covering conversion, storage sharing, variable lengths, EOG NaNs,
  incompatible batches, custom batch pinning, and CPU transfer.
- Created and executed `notebooks/3.0-torch-dataset-gpu.ipynb` on one canonical
  `Data_Train/exec` sample and one canonical `Data_Pattern/patt` sample.
- Verified a pinned batch with raw lengths 16,001 and 26,001, non-blocking CUDA transfer, and a
  finite `Conv1d` forward/backward pass with finite parameter gradients.
- Ruff passed and the full suite reported 129 passed. Stop for user review before checkpoint 3.

### 3. Preprocessed PyTorch Dataset - Pending

- Implement `TorchPreprocessedDataset` over all four spectral dataset classes.
- Implement FFT stacking and TFR time-axis padding with independent spectral/EOG masks.
- Test all methods, incompatible batches, padding, pinning, and device transfer.
- Create and execute `notebooks/3.1-torch-preprocessed-dataset-gpu.ipynb`, including finite CUDA
  `Conv2d` forward/backward checks for small canonical batches.
- Run Ruff and the full test suite, then stop for user review.

### 4. Integration - Pending

- Add automated checks that both notebooks are fully executed, error-free, use CUDA, and include
  `Data_Train/exec` and `Data_Pattern/patt`.
- Record final contracts and verification results in the memory bank.
- Run final Ruff and full pytest verification.

## Acceptance Constraints

- Zero is the padding value; masks distinguish padding from observed samples.
- Original EOG NaNs are preserved and represented by a separate finite-value mask.
- Notebooks use only small canonical batches and do not warm the full corpus.
- Prediction targets, normalization, fixed windows, and leakage-safe evaluation splits remain
  outside this phase.
- Implementation stops after every checkpoint for user review.
