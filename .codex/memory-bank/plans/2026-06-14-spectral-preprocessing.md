# EEG Spectral Preprocessing Plan

Status: active
Last updated: 2026-06-14
Next checkpoint: 7 - STFT

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

### 2. Common API And Configuration - Completed

- Implemented `PreprocessedDataset` with `FFTDataset`, `MorletDataset`, `SuperletDataset`, and
  `STFTDataset`.
- Added a typed `SpectralSample` carrying power, original EOG, axes, channels, sampling rates,
  method, scaling, and source metadata.
- Added and validated `confs/preprocessing/{common,fft,morlet,superlet,stft}.yaml` with
  OmegaConf and Pydantic.
- Defaults are EEG-only transform, 125 Hz analysis rate, 2-40 Hz, `float32`, no repeated filter,
  notch, reference, or dataset-wide normalization.
- Added validation for output dimensions, finite non-negative power, exact frequency axes, method
  scaling, and non-finite source EEG.

### 3. Spectral Artifact Cache - Completed

- Stored derived entries under
  `artifacts/preprocessed/<dataset>/<family>/<method>/<config-hash>/S_*/Trial_*/Block_*/`.
- Entries contain `eeg_power.npy`, `frequencies.npy`, optional `times.npy`, and `manifest.json`.
- Original EOG is read through `NumpyDataset` rather than duplicated.
- Writes are atomic and entries are invalidated on either source FIF, resolved config, source dtype,
  schema, transform class, or transform-version changes.
- Incomplete, corrupt, or structurally inconsistent entries are rebuilt automatically.

### 4. FFT - Completed

- Implemented `scipy.signal.resample_poly`, `scipy.fft.rfft`, and `rfftfreq` with channel-wise
  demeaning and a periodic Hann window.
- Produced correctly scaled one-sided density PSD and power-preserving overlap rebinning onto the
  exact 2-40 Hz, 1 Hz grid.
- Created and executed `notebooks/2.1-fft.ipynb` on synthetic tones and one canonical block from
  each recording family.
- Tested peak frequency, integrated PSD scaling, demeaning, validation, shape, dtype, indexing, and
  cache reuse.
- Verified the complete repository with Ruff and 97 passing tests.

### 5. Morlet - Completed

- Implemented `mne.time_frequency.tfr_array_morlet` with `output="power"`, `zero_mean=True`,
  `use_fft=True`, and `decim=1`.
- Used `n_cycles=clip(freq / 2, 3, 10)`, trimmed each side by half the longest actual wavelet, and
  averaged centered power over 32 samples for a 0.256 s time step.
- Created and executed `notebooks/2.2-morlet.ipynb` on synthetic bursts and one canonical block from
  each recording family.
- Tested cycle construction, frequency and timing localization, edge trimming, time axes, validation,
  dataset integration, and cache reuse.
- Verified the complete repository with Ruff and 101 passing tests.

### 6. Superlet - Completed

- Reworked the existing fractional adaptive implementation into a typed, validated module while
  preserving its coefficients to numerical precision.
- Documented the exact `tensionhead/Superlets` source revision and copied its MIT license notice.
- Used adaptive order 1-10 with `c_1=3`, stored `abs(coefficients) ** 2`, trimmed 199 samples per
  side from the longest contributing fractional-order wavelet, and averaged centered 32-sample bins.
- Created and executed `notebooks/2.3-superlet.ipynb` on a synthetic 20/24 Hz pair and one canonical
  block from each recording family.
- Tested adaptive orders, shape, finite non-negative output, close-frequency separation, edge/time
  axes, validation, dataset integration, and cache reuse.
- Verified the complete repository with Ruff and 105 passing tests.

Handoff state: checkpoints 1-6 are complete. Continue in a new chat from checkpoint 7 without
reimplementing or revalidating Superlet unless its contract is changed.

### 7. STFT - Pending

- Use `scipy.signal.ShortTimeFFT.from_window`.
- Use a periodic 2 s Hann window, 32-sample hop, `mfft=250`, `fft_mode="onesided2X"`, and
  PSD scaling.
- Exclude padded border slices through `lower_border_end` and `upper_border_begin`.
- Create and execute `notebooks/2.4-stft.ipynb`.
- Test axes, PSD scaling, and synthetic burst localization.

### 8. Comparative Visualization Notebook - Pending

- Create and execute `notebooks/2.5-spectral-methods-comparison.ipynb` after all four transforms
  are implemented.
- Apply FFT, Morlet, Superlet, and STFT to the same deterministic synthetic signal, one `exec`
  block, and one `patt` block.
- Show the source trace, global FFT PSD, Morlet/Superlet/STFT time-frequency maps, frequency
  marginals, time marginals, output shapes, axis resolution, runtime, and artifact size.
- Use the same EEG channel and time interval across methods. Keep native values available in
  method-specific panels.
- For shared visual panels, use explicitly labelled per-method display normalization because
  PSD and wavelet power have different scales; never treat normalized display values as stored
  ML features.
- Validate known synthetic frequencies and burst intervals before interpreting real-data plots.

### 9. Integration - Pending

- Enforce FFT shape `(channel, frequency)` and TFR shape `(channel, frequency, time)`.
- Validate one `exec` and one `patt` block in every method notebook and the comparison notebook
  without processing the full corpus.
- Run `uv run ruff check .` and `uv run pytest`.
- Update the memory bank and estimate full artifact storage.

## Scientific Constraints

- `morelet` is corrected to `morlet`.
- EOG remains an original auxiliary signal; its NaNs are permitted and binary event channels are
  not transformed as EEG.
- Non-finite EEG is an error.
- FFT/STFT use PSD scaling; Morlet/Superlet use wavelet power. Their absolute values are not
  directly comparable.
- Cross-method plot normalization is presentation-only and must not be written back into artifacts.
- `exec` and `patt` are recording families, not an inferred train/test split.
- Implementation stops after each checkpoint for review.
