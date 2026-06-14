# Experiments

## 2026-06-14 - FeatureDataset cache and sklearn export validation

- Artifact: executed `notebooks/4.2-feature-dataset-export.ipynb`.
- Data: canonical key `(1, 1, 1)` from `Data_Train/exec` and `Data_Pattern/patt`, kept as separate
  datasets and exports.
- Configuration: crop `[0.5, 15.5)`, 125 Hz analysis rate, six complete 5 s windows with 2 s
  stride, default `float32` output.
- Cache check: each family was first extracted into an isolated temporary modular cache. A second
  `FeatureDataset` loaded the same arrays while its source-array loader was configured to raise,
  confirming that valid feature-cache hits do not materialize source EEG/EOG.
- Export: selecting `time`, `spectral`, and `lndp` produced shape `(6, 17829)` per family:
  819 time columns, 882 spectral columns, and 16,128 LNDP columns. Every row retained parent key
  `(1, 1, 1)`, window index `0..5`, and absolute bounds from `[0.5, 5.5)` through `[10.5, 15.5)`.
- Verification: seven code cells executed without errors, emitted
  `FEATURE_DATASET_EXPORT_VERIFIED`, and both charts were visually inspected. Ruff passed and the
  full suite reported 211 passed and 2 skipped.
- Interpretation boundary: timings are illustrative single-run infrastructure observations, not a
  performance benchmark. No target, split, scaling, PCA, feature selection, or model was fitted.

## 2026-06-14 - Classical and local-pattern feature validation

- Artifacts: executed `notebooks/4.0-classical-features.ipynb` and
  `notebooks/4.1-local-patterns.ipynb`.
- Synthetic classical checks: amplitude-2 10 Hz and amplitude-1 20 Hz tones recovered alpha power
  2.0 and beta power 0.5, with dominant frequencies at 10 Hz and 20 Hz. OAS covariance,
  correlation, and log-covariance were finite and symmetric.
- Article checks: the exact Fig. 3 segment produced LNDP code 7 and the exact Fig. 4 segment
  produced 1D-LGP code 224 for `m=8`.
- Local-pattern contract: complete neighborhoods only, 256-bin per-channel histograms, raw-count
  support, and L1 probability mode by default. Synthetic codes were unchanged by a global offset.
- Real data: canonical `Data_Pattern/patt` key `(1, 1, 1)` produced finite full-epoch blocks with
  shapes `(1, 63, 13)`, `(1, 63, 14)`, three `(1, 63, 63)` matrices, and three
  `(1, 63, 256)` local-pattern histograms. Five-second windows with 2 s stride produced six
  windows for every family.
- Verification: both notebooks executed without errors and were visually inspected;
  `uv run ruff check .` passed and `uv run pytest` reported 202 passed and 2 skipped.
- Interpretation boundary: this validates extraction mechanics and transfer to multi-channel
  imagery EEG only. It does not reproduce the article's seizure-classification accuracy and does
  not evaluate any predictive model.

## 2026-06-14 - FFT checkpoint validation

- Code/config: `preprocessors/fft.py`, `FFTDataset`, default `confs/preprocessing/fft.yaml`, and
  executed `notebooks/2.1-fft.ipynb`.
- Data: deterministic synthetic tones plus canonical key `(1, 1, 1)` from both `Data_Train/exec`
  and `Data_Pattern/patt`; no ML split or model was involved.
- Preprocessing: resample 1,000 Hz to 125 Hz, channel-wise demean, periodic Hann, one-sided density
  periodogram, overlap rebinning to 2-40 Hz in 1 Hz steps.
- Synthetic results: 10 Hz and 23 Hz peaks were recovered exactly. Integrated-power relative errors
  were 0.0722% and 0.0091%, respectively.
- Real outputs: both blocks produced finite non-negative `float32` arrays with shape `(63, 39)` and
  no time axis. Spectral cache entries were about 12 KiB per block and contained no EOG array.
- Verification: `uv run ruff check .` passed and `uv run pytest` reported 97 passed.
- Interpretation boundary: this validates transform mechanics only; it does not compare conditions,
  subjects, labels, or predictive performance.

## 2026-06-14 - Morlet checkpoint validation

- Code/config: `preprocessors/morlet.py`, `MorletDataset`, default
  `confs/preprocessing/morlet.yaml`, and executed `notebooks/2.2-morlet.ipynb`.
- Data: deterministic synthetic 10 Hz and 25 Hz bursts plus canonical key `(1, 1, 1)` from both
  `Data_Train/exec` and `Data_Pattern/patt`; no ML split or model was involved.
- Preprocessing: resample 1,000 Hz to 125 Hz, zero-mean Morlet power with
  `n_cycles=clip(frequency / 2, 3, 10)`, trim 149 samples per side, and average centered
  non-overlapping 32-sample bins.
- Synthetic results: both frequencies were recovered exactly. Peak-time center errors were 0.052 s
  for the 10 Hz burst and 0.044 s for the 25 Hz burst.
- Real outputs: `exec` produced `(63, 39, 53)` and `patt` produced `(63, 39, 92)` `float32` arrays
  with a 0.256 s time step. Cache entries were about 512 KiB and 886 KiB, respectively.
- Verification: `uv run ruff check .` passed and `uv run pytest` reported 101 passed.
- Interpretation boundary: this validates time-frequency localization and storage contracts only;
  display-normalized dB values are not cached or treated as ML features.

## 2026-06-14 - Superlet checkpoint validation

- Code/config: `preprocessors/superlet.py`, `SuperletDataset`, default
  `confs/preprocessing/superlet.yaml`, and executed `notebooks/2.3-superlet.ipynb`.
- Provenance: typed adaptation of `tensionhead/Superlets` commit
  `20f6bfdf31b783b4d8254546effa8f27784118a2` with the upstream MIT license notice retained.
- Data: deterministic synthetic stationary 20 Hz and 24 Hz tones plus canonical key `(1, 1, 1)`
  from both `Data_Train/exec` and `Data_Pattern/patt`; no ML split or model was involved.
- Preprocessing: resample 1,000 Hz to 125 Hz, fractional adaptive order 1-10 with `c_1=3`, store
  coefficient magnitude squared, trim 199 samples per side, and average centered non-overlapping
  32-sample bins.
- Synthetic results: 20 Hz and 24 Hz formed separate local maxima with mean powers 0.209 and 0.211;
  the 22 Hz midpoint fell to 0.075. Output was finite, non-negative, and shaped `(1, 39, 26)`.
- Real outputs: `exec` produced `(63, 39, 50)` and `patt` produced `(63, 39, 89)` `float32` arrays
  with a 0.256 s time step. Cache entries were about 483 KiB and 857 KiB, respectively.
- Verification: the refactored transform matched the previous implementation within about `1.2e-8`
  in complex coefficients; Ruff passed and `uv run pytest` reported 105 passed.
- Interpretation boundary: this validates close-frequency resolution and storage contracts only;
  display-relative dB values are not cached or treated as ML features.

## 2026-06-14 - STFT checkpoint validation

- Code/config: `preprocessors/stft.py`, `STFTDataset`, default `confs/preprocessing/stft.yaml`, and
  executed `notebooks/2.4-stft.ipynb`.
- Data: deterministic synthetic 10 Hz and 25 Hz bursts, a stationary 10 Hz sine, and canonical key
  `(1, 1, 1)` from both `Data_Train/exec` and `Data_Pattern/patt`; no ML split or model was involved.
- Preprocessing: resample to 125 Hz, periodic 2 s Hann, 32-sample hop, `mfft=250`,
  `fft_mode="onesided2X"`, PSD scaling, exclusion of padded border slices, and power-preserving
  overlap rebinning from the native 0.5 Hz grid to 2-40 Hz in 1 Hz steps.
- Synthetic results: both bursts were recovered at the correct frequencies and inside their
  generating intervals. Integrated PSD of an amplitude-2 stationary sine equaled its expected
  mean square of 2.0 for every retained time slice.
- Real outputs: `exec` produced `(63, 39, 55)` and `patt` produced `(63, 39, 94)` `float32` arrays
  with a 0.256 s time step. Cache entries were about 531 KiB and 905 KiB, respectively.
- Verification: `uv run ruff check .` passed and `uv run pytest` reported 110 passed.
- Interpretation boundary: this validates time-frequency localization, density scaling, border
  handling, and storage contracts only; logarithmic display values are not cached features.

## 2026-06-14 - Spectral methods comparison

- Artifact: executed `notebooks/2.5-spectral-methods-comparison.ipynb`.
- Data: one shared synthetic signal with 10 Hz and 25 Hz bursts plus `Fp1`, 2-14 s, from canonical
  key `(1, 1, 1)` in both `Data_Train/exec` and `Data_Pattern/patt`.
- Synthetic validation: FFT recovered 10 Hz and 25 Hz as its two largest bins. Morlet, Superlet,
  and STFT recovered each generating frequency at the burst center and placed peak time inside the
  correct generating interval.
- Shared axes: every method returned 2-40 Hz in 1 Hz steps at 125 Hz analysis rate. Morlet,
  Superlet, and STFT each used a 0.256 s output step but retained different valid time supports.
- Illustrative single-channel timings on this machine were roughly 1 ms for FFT, 3-5 ms for
  Morlet, 2-3 ms for STFT, and 36-52 ms for Superlet on the canonical real blocks. These timings
  exclude cache I/O and are not a scientific quality ranking.
- Full cached artifact sizes remained about 12 KiB for FFT, 512/886 KiB for Morlet, 483/857 KiB
  for Superlet, and 531/905 KiB for STFT on the demonstrated `exec`/`patt` blocks.
- Verification: all seven code cells executed without warnings or errors; `uv run ruff check .`
  passed and `uv run pytest` reported 110 passed.
- Interpretation boundary: native PSD and wavelet-power amplitudes were shown only in
  method-specific panels. Shared relative-dB maps and peak-normalized time marginals were
  presentation-only and were not written to artifacts.

## 2026-06-14 - Spectral preprocessing integration

- Scope: final checkpoint across FFT, Morlet, Superlet, STFT, notebooks `2.1` through `2.5`, and
  canonical key `(1, 1, 1)` in both recording families.
- Code contract: `PreprocessedDataset` now rejects 3D FFT output and rejects 2D Morlet, Superlet,
  or STFT output. The common transform schema continues to validate either representation before
  the method-specific dataset boundary is applied.
- Canonical artifact validation: all eight entries loaded through public dataset classes with the
  expected shapes, `float32` dtype, exact 2-40 Hz grid, correct PSD/wavelet-power scaling, and no
  cached `eog.npy`.
- Notebook validation: all code cells in `notebooks/2.1-fft.ipynb` through
  `notebooks/2.5-spectral-methods-comparison.ipynb` have execution counts, no error outputs, and
  explicit `Data_Train/exec` plus `Data_Pattern/patt` demonstrations.
- Corpus duration groups used for storage estimation: 1,200 `exec` blocks at 16.000 s, 59 at
  15.499 s, one at 15.414 s, and 540 `patt` blocks at 26.000 s.
- Estimated logical artifact sizes: FFT 22,120,560 bytes (21.10 MiB), Morlet 1,148,894,408 bytes
  (1.070 GiB), Superlet 1,095,727,808 bytes (1.020 GiB), and STFT 1,184,183,408 bytes
  (1.103 GiB). Total: 3,450,926,184 bytes, about 3.214 GiB or 3.451 GB.
- Estimate assumptions: 63 channels, 39 frequencies, `float32`, actual duration-group time-bin
  counts, `.npy` headers, and observed canonical manifest sizes. Filesystem block and directory
  allocation overhead are excluded, so provision slightly more than 3.45 GB.
- Verification: `uv run ruff check .` passed and `uv run pytest` reported 117 passed.

## 2026-06-14 - Raw PyTorch dataset CUDA verification

- Scope: `TorchDataset`, raw variable-length collation, and
  `notebooks/3.0-torch-dataset-gpu.ipynb`.
- Inputs: canonical block `(1, 1, 1)` from `Data_Train/exec` and `Data_Pattern/patt`.
- Batch result: EEG shape `(2, 63, 26001)` with observed lengths 16,001 and 26,001 samples;
  shorter-sample padding was zero and the valid-time mask reproduced both lengths.
- EOG result: source NaNs were preserved, padding was excluded, and `eog_finite_mask` matched
  `isfinite` over every observed sample.
- GPU result: custom DataLoader batch tensors were pinned, non-blocking transfer reached CUDA,
  and a small `Conv1d` model completed forward/backward with finite output, loss, and gradients.
- Interpretation boundary: this is an infrastructure smoke test, not a trained model or scientific
  performance result. No target or data split was introduced.
- Verification: all four notebook code cells executed without errors and emitted
  `CUDA_VERIFIED`; Ruff passed and the full suite reported 129 passed.

## 2026-06-14 - Spectral PyTorch datasets CUDA verification

- Scope: `TorchPreprocessedDataset`, method-aware spectral collation, and
  `notebooks/3.1-torch-preprocessed-dataset-gpu.ipynb`.
- Inputs: canonical block `(1, 1, 1)` from both `Data_Train/exec` and `Data_Pattern/patt` for FFT,
  Morlet, Superlet, and STFT, loaded through existing validated spectral caches.
- FFT batch: power shape `(2, 63, 39)` with no spectral time metadata.
- Morlet batch: padded power shape `(2, 63, 39, 92)` with spectral lengths 53 and 92.
- Superlet batch: padded power shape `(2, 63, 39, 89)` with spectral lengths 50 and 89.
- STFT batch: padded power shape `(2, 63, 39, 94)` with spectral lengths 55 and 94.
- GPU result: all custom batches were pinned, transferred non-blocking to CUDA, and completed a
  method-specific `Conv2d` forward/backward pass with finite outputs, losses, and gradients.
- Interpretation boundary: these are infrastructure checks over cached features, not learned models
  or evidence that one spectral representation is scientifically superior.
- Verification: all four notebook code cells executed without errors, covered all methods and both
  recording families, and emitted `CUDA_VERIFIED`; Ruff passed and the full suite reported
  150 passed.

## 2026-06-14 - PyTorch dataset integration

- Scope: public raw and spectral PyTorch dataset APIs, both executed CUDA notebooks, and automated
  notebook acceptance checks.
- Notebook contract: every code cell has an execution count, no output is an error,
  `torch.cuda.is_available()` is required, the selected device is CUDA, and stored outputs contain
  `CUDA_VERIFIED`.
- Coverage: both notebooks include `Data_Train/exec` and `Data_Pattern/patt`; the raw notebook
  exercises `TorchDataset`, raw collation, and `Conv1d`; the spectral notebook exercises
  `TorchPreprocessedDataset`, spectral collation, `Conv2d`, and all four spectral methods.
- Public API validation: raw/spectral sample and batch schemas, both adapters, and both collators
  import successfully from `utils.datasets`.
- Interpretation boundary: notebook tests prove the stored infrastructure demonstrations ran on
  CUDA; they do not establish predictive validity, model quality, or an evaluation protocol.
- Verification: Ruff passed and the full suite reported 152 passed. Two Python 3.13
  multiprocessing `fork()` deprecation warnings remain in pre-existing disk-cache tests.

## 2026-06-14 - Integrated EEG feature validation

- Artifact: executed `notebooks/4.3-scientific-feature-validation.ipynb`.
- Inputs: deterministic 15 s synthetic tones and correlated channels plus canonical
  `Data_Pattern/patt` key `(1, 1, 1)`, cropped to `[0.5, 15.5)` and resampled to 125 Hz.
- Temporal comparison: one full 15 s epoch versus six complete overlapping 5 s windows with a
  2 s stride. Every window retains the same parent block and is not an independent split unit.
- Synthetic results: amplitude-2 10 Hz alpha power `2.001444`, amplitude-1 20 Hz beta power
  `0.501091`, and designed alpha-channel correlation `0.994901`.
- Real-data results: Fp1 full-crop alpha power `0.00043554`; window range
  `0.00023185-0.00098232`; mean absolute full/window-average correlation difference `0.011463`;
  maximum Fp1 LNDP L1 distance from the full histogram `0.395448`.
- Unit caveat: MNE reports source EEG units as volts, while the canonical Fp1 crop spans
  0.665228 V peak-to-peak, which is atypically large for physiological EEG. No unit correction,
  normalization, or physiological amplitude interpretation was introduced.
- Local-pattern contract: probability histograms sum to one and the paper examples reproduce
  LNDP code `7` and 1D-LGP code `224`.
- Visual QA: signal/band-power, synthetic correlation, real correlation, and LNDP figures were
  inspected after final execution; labels, scales, legends, and stated grouping boundaries were
  readable and consistent.
- Interpretation boundary: extraction validation only. No target, model, learned transform,
  split, condition effect, classification result, or reproduction of article accuracy is claimed.
- Verification: eight code cells executed with no errors, four PNG outputs were stored, Ruff
  passed, `git diff --check` passed, and the full suite reported 212 passed and 2 skipped.
