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

Spectral preprocessing checkpoints 1-8 are complete:

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
- The current implementation plan is stored in
  `.codex/memory-bank/plans/2026-06-14-spectral-preprocessing.md`.
- Checkpoint 8 is complete; the next chat should start directly with checkpoint 9 (integration).

## Next Actions

- Complete checkpoint 9 integration validation and estimate full-corpus artifact storage.
- Reconfirm every method notebook and the comparison notebook against one canonical `exec` and
  `patt` block without warming the full corpus.
- Define canonical train/validation/test split policy.
- Decide how labels from `labels.json` map to targets.
- Benchmark full-corpus cache warmup only when operational timing is needed.

## Open Questions

- What is the exact prediction target?
- Should evaluation be leave-one-subject-out, grouped K-fold by subject, within-subject, or multiple protocols?
- Which recordings are considered training, pattern/reference, validation, or test data?
