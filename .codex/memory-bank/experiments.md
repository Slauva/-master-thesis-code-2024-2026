# Experiments

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
