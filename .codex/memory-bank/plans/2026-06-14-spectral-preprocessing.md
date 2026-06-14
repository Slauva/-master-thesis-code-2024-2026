# EEG Spectral Preprocessing Plan

Status: completed
Last updated: 2026-06-14
Next checkpoint: complete

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

Handoff state: checkpoints 1-9 are complete. The next project phase should define prediction
targets and leakage-safe evaluation splits before model development.

### 7. STFT - Completed

- Implemented `scipy.signal.ShortTimeFFT.from_window` with a periodic 2 s Hann window,
  32-sample hop, `mfft=250`, `fft_mode="onesided2X"`, and PSD scaling.
- Excluded padded border slices through `lower_border_end` and `upper_border_begin`.
- Reused power-preserving density overlap rebinning to map the native 0.5 Hz STFT grid onto the
  exact 2-40 Hz, 1 Hz project grid.
- Created and executed `notebooks/2.4-stft.ipynb` on synthetic bursts, a stationary scaling check,
  and one canonical block from each recording family.
- Tested axes, minimum valid signal length, PSD integration, burst localization, dataset
  integration, and cache reuse.
- Verified the complete repository with Ruff and 110 passing tests.

### 8. Comparative Visualization Notebook - Completed

- Created and executed `notebooks/2.5-spectral-methods-comparison.ipynb`.
- Applied all four methods to the same deterministic 10 Hz/25 Hz burst signal and canonical
  `(1, 1, 1)` blocks from both recording families.
- Included source traces, global FFT PSD, all three time-frequency maps, native frequency
  marginals, display-normalized time marginals, output shapes, axis resolution, direct
  single-channel runtime, and full cached artifact size.
- Used `Fp1` and the common 2-14 s interval for every real-data method panel.
- Kept native PSD and wavelet-power values in method-specific panels; shared relative-dB maps and
  peak-normalized curves are explicitly labelled as presentation-only.
- Validated FFT global peaks and all time-frequency burst locations before plotting real data.
- Verified the complete repository with Ruff and 110 passing tests.

### 9. Integration - Completed

- Enforced FFT shape `(channel, frequency)` and TFR shape
  `(channel, frequency, time)` at the dataset boundary.
- Added automated checks that every method notebook and the comparison notebook are executed,
  error-free, and include both canonical recording families.
- Revalidated one `exec` and one `patt` cache entry for every method through the public dataset
  classes without processing the full corpus.
- Estimated logical storage for all four methods across 1,800 blocks at about 3.214 GiB
  (3.451 GB), excluding filesystem allocation overhead.
- Verified the complete repository with Ruff and 117 passing tests.

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
