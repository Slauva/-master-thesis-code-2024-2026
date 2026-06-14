# EEG Spectral Preprocessing Plan

Status: active
Last updated: 2026-06-14

## Documentation Rule

Before using a new library API:

1. Check the official library documentation through Context7.
2. If Context7 is incomplete, inspect the version installed in `.venv`.
3. Confirm ambiguous behavior with a minimal synthetic call.
4. Protect the resulting contract with tests and record durable decisions.

## Checkpoints

### 1. Dataset Overview - Completed

- Created and executed `notebooks/2.0-dataset-overview.ipynb`.
- Audited FIF metadata for all 1,800 canonical blocks.
- Computed PSD, EOG quality summaries, and relative band-power topographies on a documented
  deterministic 16-block sample.
- Confirmed 1,000 Hz input, 63 EEG channels, 5 EOG channels, and a stored EEG passband of 1-40 Hz.
- Found 60 shortened `exec` blocks and missing `EOG_x`/`EOG_y` intervals in sampled `patt` blocks.

### 2. Common API And Configuration - Pending

- Implement `PreprocessedDataset` with `FFTDataset`, `MorletDataset`, `SuperletDataset`, and
  `STFTDataset`.
- Add a typed `SpectralSample` carrying power, original EOG, axes, channels, sampling rates,
  method, scaling, and source metadata.
- Load and validate `confs/preprocessing/{common,fft,morlet,superlet,stft}.yaml` with
  OmegaConf and Pydantic.
- Defaults: EEG-only transform, 125 Hz analysis rate, 2-40 Hz, `float32`, no repeated filter,
  notch, reference, or dataset-wide normalization.

### 3. Spectral Artifact Cache - Pending

- Store derived entries under
  `artifacts/preprocessed/<dataset>/<family>/<method>/<config-hash>/S_*/Trial_*/Block_*/`.
- Save `eeg_power.npy`, `frequencies.npy`, optional `times.npy`, and `manifest.json`.
- Read original EOG through `NumpyDataset` rather than duplicating it.
- Use atomic writes and invalidate on source FIF, config, schema, or transform-version changes.

### 4. FFT - Pending

- Use `scipy.fft.rfft` and `rfftfreq` with channel-wise demeaning and a periodic Hann window.
- Produce correctly scaled one-sided PSD and aggregate bins onto a 2-40 Hz, 1 Hz grid.
- Create and execute `notebooks/2.1-fft.ipynb`.
- Test peak frequency, PSD scaling, shape, dtype, indexing, and cache behavior.

### 5. Morlet - Pending

- Use `mne.time_frequency.tfr_array_morlet` with `output="power"`, `zero_mean=True`, and
  `decim=1`.
- Use `n_cycles=clip(freq / 2, 3, 10)`, trim wavelet-dependent edges, then average power over
  32 samples for a 0.256 s time step.
- Create and execute `notebooks/2.2-morlet.ipynb`.
- Test synthetic burst frequency and timing localization.

### 6. Superlet - Pending

- Type and validate the existing implementation and document its provenance/license.
- Use adaptive Superlet with `order_min=1`, `order_max=10`, and `c_1=3`.
- Store `abs(coefficients) ** 2`, trim edges, and aggregate to a 0.256 s time step.
- Create and execute `notebooks/2.3-superlet.ipynb`.
- Test shape, finite non-negative output, and close-frequency separation.

### 7. STFT - Pending

- Use `scipy.signal.ShortTimeFFT.from_window`.
- Use a periodic 2 s Hann window, 32-sample hop, `mfft=250`, `fft_mode="onesided2X"`, and
  PSD scaling.
- Exclude padded border slices through `lower_border_end` and `upper_border_begin`.
- Create and execute `notebooks/2.4-stft.ipynb`.
- Test axes, PSD scaling, and synthetic burst localization.

### 8. Integration - Pending

- Enforce FFT shape `(channel, frequency)` and TFR shape `(channel, frequency, time)`.
- Validate one `exec` and one `patt` block in every notebook without processing the full corpus.
- Run `uv run ruff check .` and `uv run pytest`.
- Update the memory bank and estimate full artifact storage.

## Scientific Constraints

- `morelet` is corrected to `morlet`.
- EOG remains an original auxiliary signal; its NaNs are permitted and binary event channels are
  not transformed as EEG.
- Non-finite EEG is an error.
- FFT/STFT use PSD scaling; Morlet/Superlet use wavelet power. Their absolute values are not
  directly comparable.
- `exec` and `patt` are recording families, not an inferred train/test split.
- Implementation stops after each checkpoint for review.
