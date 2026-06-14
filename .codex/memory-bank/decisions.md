# Decisions

## 2026-06-10

- Store project operating instructions in `Agent.md`.
- Store persistent project context in the project memory bank.
- Store project-local skills under `.codex/skills/`.
- Keep `python_optimization_prompt.md` content in `.codex/skills/python-optimization-panel/references/python_optimization_prompt.md` so the prompt survives deletion of the root file.
- Keep `optimization_methods.pdf` and a compact markdown summary in `.codex/skills/python-optimization-panel/references/` so the optimization handout survives deletion of the root file.
- Treat leakage control as a first-class scientific constraint for all EEG/ML work.

## 2026-06-14

- Keep dataset indexing and metadata in `DatasetBase`; keep FIF loading and array caches in `NumpyDataset`.
- Treat FIF files as the source of truth and invalidate derived cache entries from source size and modification time.
- Use atomic per-sample `.npy` disk-cache writes plus an optional process-local bounded LRU.
- Make cache warmup explicit through `warm_cache()`; constructors must not start heavy preload work.
- Use process workers only for disk-cache generation. Parent-process RAM cache remains local and is populated by normal sample access.
- Keep reusable EEG dataset and experiment semantics in the standalone
  `eeg-dataset-ml-experiments-semantic-layer`; do not infer a scientific train/test split from
  `Data_Train` and `Data_Pattern` names.
- Build spectral datasets as typed wrappers over `NumpyDataset`; preserve the canonical sample key and
  original EOG while transforming EEG only.
- Compose `confs/preprocessing/common.yaml` with one method YAML through OmegaConf, then validate the
  resolved mapping with frozen Pydantic models that reject extra or unsupported fields.
- Require every spectral transform to return the exact inclusive frequency grid defined by
  `f_min`, `f_max`, and `frequency_step`.
- Keep repeated filtering, notch filtering, rereferencing, EOG transformation, and dataset-wide
  normalization disabled in the baseline spectral configs.
- Compare FFT, Morlet, Superlet, and STFT in a dedicated executed notebook using identical synthetic
  and real inputs. Any cross-method display normalization is presentation-only because PSD and
  wavelet-power amplitudes are not directly comparable.
- Store spectral cache entries under
  `artifacts/preprocessed/<dataset>/<family>/<method>/<config-hash>/S_*/Trial_*/Block_*/`.
- Cache only `eeg_power.npy`, `frequencies.npy`, optional `times.npy`, and `manifest.json`; retrieve
  current EOG from `NumpyDataset` instead of duplicating it in spectral artifacts.
- Include resolved preprocessing config, source dtype, schema version, transform class, and transform
  version in spectral cache identity. Validate both EEG and EOG source signatures on every cache hit.
- Use atomic per-array writes and write the spectral manifest last; treat incomplete or corrupt entries
  as disposable and rebuild them.
- Store the project memory bank under `.codex/memory-bank/`; this supersedes its original root-level
  location.
- Resample EEG to the configured FFT analysis rate with `scipy.signal.resample_poly` before spectral
  estimation; preserve original EOG and its source sampling rate separately.
- Compute FFT density PSD as `abs(rfft(x * window)) ** 2 / (fs * sum(window ** 2))`, doubling only
  one-sided bins that have negative-frequency partners.
- Rebin native FFT density bins onto configured output frequencies by overlap between frequency-cell
  edges. Divide integrated overlap power by the output-bin width so band power is preserved.
- Use MNE Morlet power with `n_cycles=clip(frequency / 2, 3, 10)`, `zero_mean=True`, `use_fft=True`,
  and `decim=1`; perform time reduction only after convolution and edge trimming.
- Define the common Morlet edge trim from half the longest actual discrete wavelet returned by MNE.
  For the default 125 Hz, 2-40 Hz configuration this is 149 samples per side.
- Center the largest complete set of non-overlapping 32-sample bins within the valid trimmed Morlet
  interval. Store bin-center times derived from actual sample indices.
- Keep the project Superlet core as a typed adaptation of Gregor Mönke's implementation from
  `tensionhead/Superlets` commit `20f6bfdf31b783b4d8254546effa8f27784118a2`; retain the copied MIT
  license notice and upstream attribution beside the module.
- Use fractional adaptive Superlet order 1-10 with `c_1=3` and store coefficient magnitude squared
  as wavelet power.
- Define the common Superlet edge trim from the longest integer-order wavelet that contributes to
  each fractional order. For the default grid, the limiting six-cycle wavelet at 3 Hz has 398
  samples of support, producing a 199-sample trim per side.
- Share centered edge-trimmed 32-sample power binning between Morlet and Superlet so both methods
  derive time coordinates from the same sample-index contract.
- Do not make cache-path tests depend on whether generated artifacts already exist; notebook
  execution and normal dataset access may legitimately populate default cache locations.
- Compute STFT PSD with `scipy.signal.ShortTimeFFT.from_window`, a periodic 2 s Hann window,
  32-sample hop, `mfft=250`, `fft_mode="onesided2X"`, and `scale_to="psd"`.
- Retain only STFT slices from `lower_border_end[1]` through
  `upper_border_begin(n_samples)[1]`; padded border slices must not enter stored features.
- Rebin STFT density from its native 0.5 Hz grid to the exact 2-40 Hz, 1 Hz output grid by
  frequency-cell overlap. Share the power-preserving density rebinning implementation with FFT;
  selecting every second native bin is invalid because it does not preserve integrated power.
- Require `window_seconds * analysis_sfreq` to be an integer number of samples in STFT configs.
- Enforce method-specific output dimensionality at the `PreprocessedDataset` boundary: FFT must
  return `(channel, frequency)`, while Morlet, Superlet, and STFT must return
  `(channel, frequency, time)`.
- Treat full-corpus spectral storage calculations as logical-size estimates based on actual corpus
  duration groups, transform time-axis formulas, NumPy array headers, and observed manifest sizes.
  Report filesystem allocation overhead separately rather than implying byte-exact disk usage.
- Keep PyTorch datasets as zero-copy map-style adapters over configured `NumpyDataset` and
  `PreprocessedDataset` instances. They must not duplicate FIF loading, preprocessing, or cache
  configuration.
- Keep individual dataset samples on CPU and move only collated batches to accelerators. Custom
  batch classes own pinned-memory and device-transfer behavior while source metadata remains on CPU.
- Pad raw EEG/EOG and spectral time axes with zeros and provide explicit lengths and masks.
  Preserve source EOG NaNs and distinguish them from padding through a separate finite-value mask.
- Stack FFT as `(batch, channel, frequency)` without time metadata. Pad Morlet, Superlet, and STFT
  as `(batch, channel, frequency, time)` and store their padded per-sample time coordinates as
  `(batch, time)` with a spectral mask.
- Reject mixed-method or scientifically incompatible spectral batches, including mismatched
  scaling, channels, sampling rates, dtypes, or frequency grids.
- Use the project-local `manage-staged-plans` skill for substantial staged implementation or
  research plans.
- Persist a new plan under `.codex/memory-bank/plans/` only after explicit user approval.
- Stop after every implemented stage for user review. Mark the stage completed only after explicit
  approval, and update the plan plus the smallest relevant memory files before continuing.
- Use the `data-analytics:jupyter-notebooks` workflow for reproducible visualization of
  quantitative or scientific stage results; execute notebooks top-to-bottom before treating their
  outputs as evidence.
- Configure EEG feature extraction through `confs/features/default.yaml`, resolve it with
  OmegaConf, and validate it with a frozen Pydantic model that rejects unsupported fields.
- Apply the canonical imagery crop as the half-open source interval `[0.5, 15.5)` before
  resampling. Require crop and optional window boundaries to resolve to exact integer samples.
- Treat `window_seconds=None` as one full-crop window. When windowing is enabled, require both
  window length and stride, retain complete windows only, and never pad a partial trailing window.
- Store extracted features as modular three-dimensional blocks: `(window, channel, feature/code)`
  for per-channel values and histograms, or `(window, channel, channel)` for spatial matrices.
- Flatten modular feature blocks in explicit block, channel, then feature/code order. Vectorize
  symmetric channel matrices from the upper triangle and multiply off-diagonal entries by
  `sqrt(2)` to preserve the Frobenius inner product.
- Include the resolved feature configuration plus cache-schema and extractor versions in a
  deterministic feature cache identity. Keep generated feature artifacts under
  `artifacts/features/`.
- Compute feature working arrays in float64 after applying the source-rate imagery crop and
  polyphase resampling; cast only completed feature blocks to the configured output dtype.
- Define normalized line length as the mean absolute first difference and zero-crossing rate as
  the proportion of adjacent raw samples whose sign bit changes.
- Compute skewness and excess kurtosis from population central moments. Return zero for undefined
  ratios on constant signals, including Hjorth mobility/complexity.
- Derive feature band powers from the established one-sided Hann density FFT on a 1 Hz grid.
  Integrate frequency-cell overlap against half-open band intervals, normalize relative powers by
  total configured-band-range power, and report normalized spectral entropy.
- Estimate each window's spatial covariance with sklearn OAS using time samples as observations
  and channels as features. Derive correlation from the OAS covariance and log-covariance through
  a symmetric eigendecomposition with a relative eigenvalue floor.
- Represent all-zero covariance, correlation, and log-covariance as finite zero matrices rather
  than emitting undefined values.
- Implement LNDP using chronological windows ordered as `P_m, ..., P_0`, including the center
  sample in consecutive differences. Assign bit zero to `P_1 - P_0`, exactly as shown in Fig. 3
  of Jaiswal and Banka (2017).
- Implement 1D-LGP and 1D-LBP over the same chronological neighborhood, excluding the center from
  the `m` neighbors and assigning bit zero to the rightmost neighbor `P_0`.
- Generate local-pattern codes only where the complete `m/2` context exists on both sides; do not
  pad signal boundaries. Use raw counts for paper-style reproduction and L1 probability
  histograms by default so different window lengths remain comparable.
- Keep `FeatureDataset` scoped to exactly one source `NumpyDataset` and one `exec` or `patt`
  recording family. Do not accept a pooled multi-family input implicitly.
- Store each extracted feature block as its own atomic `.npy` file and write the shared feature
  manifest last. Isolate cache roots by dataset name, recording family, source dtype, and the
  versioned resolved-feature-config hash.
- Invalidate feature cache entries when schema/extractor version, resolved config, source dtype,
  either FIF signature, source sampling rate, channel order, block schema, shape, or dtype differs.
- Check the feature manifest and current FIF signatures before materializing source arrays.
  Load through `NumpyDataset` only when the feature entry is absent, stale, incomplete, or corrupt.
- Export sklearn rows at window grain while repeating the canonical parent
  `(subject_id, trial_number, block_index)` key for every child window. Keep the recording family,
  zero-based window index, and absolute bounds alongside `X`.
- Keep scaling, PCA, feature selection, targets, and all learned transforms outside
  `FeatureDataset` and `build_feature_matrix(...)`; fit them only within training folds.
