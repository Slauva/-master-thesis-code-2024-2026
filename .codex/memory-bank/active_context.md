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

EEG feature extraction:

- The approved staged implementation plan is stored in
  `.codex/memory-bank/plans/2026-06-14-eeg-feature-extraction.md`.
- Stage 1, contracts and configuration, is completed.
- Stage 2, classical time, spectral, and spatial features, is completed.
- Stage 3, Jaiswal-Banka local patterns, is completed.
- Stage 4, dataset cache and sklearn export, is completed.
- Stage 5, final scientific validation, is implemented and awaiting explicit final review.
- Added strict feature configuration, exact imagery crop/window layouts, modular feature schemas,
  stable flattening, and versioned config hashing.
- Verified the default `[0.5, 15.5)` crop on canonical `exec` and `patt` samples; both produce
  `(63, 15000)` before resampling.
- Added 13 time features, band-power/spectral summaries, OAS covariance, correlation, and
  log-covariance with finite constant-signal handling.
- Canonical `Data_Pattern/patt` extraction produces finite blocks with expected 63-channel shapes.
- Added and executed `notebooks/4.0-classical-features.ipynb` with synthetic checks, spatial
  matrices, and a full-epoch versus six-window real imagery demonstration.
- Added vectorized per-channel LNDP, 1D-LGP, and 1D-LBP with count or probability histograms,
  complete neighborhoods, and the paper's exact bit order.
- Added and executed `notebooks/4.1-local-patterns.ipynb`; published examples reproduce LNDP code
  7 and 1D-LGP code 224, while the canonical imagery block produces finite `(1/6, 63, 256)`
  histograms.
- Added common `extract_feature_set(...)`, `FeatureDataset`, modular atomic feature caching, typed
  `FeatureMatrix`, and `build_feature_matrix(...)`.
- Feature cache identity covers dataset/family, source dtype, resolved config, schema/extractor
  versions, and both source FIF signatures. Every exported window retains its parent sample key,
  window index, absolute bounds, and recording family.
- Valid feature-cache hits are resolved before source EEG/EOG arrays are loaded.
- Canonical `exec` and `patt` key `(1, 1, 1)` each produce eight finite cached feature blocks;
  selected `time+spectral` export has shape `(1, 1701)`.
- Added and executed `notebooks/4.2-feature-dataset-export.ipynb`. With 5 s windows and 2 s stride,
  each family exports six rows and 17,829 `time+spectral+lndp` columns while preserving parent
  keys, indices, and bounds; a cache hit is demonstrated without source-array loading.
- Added and executed `notebooks/4.3-scientific-feature-validation.ipynb` with integrated synthetic
  and canonical `Data_Pattern/patt` validation, four visually inspected figures, one full 15 s
  crop, and six complete 5 s windows.
- Synthetic alpha/beta powers were 2.001444 and 0.501091; designed alpha correlation was 0.994901.
  Canonical Fp1 alpha power was 0.00043554 for the full crop and 0.00023185-0.00098232 across
  windows. Mean absolute correlation-grain difference was 0.011463 and maximum Fp1 LNDP L1
  distance from the full histogram was 0.395448.
- The source FIF declares volts but the canonical Fp1 crop has an atypically large 0.665 V
  peak-to-peak scale. Feature extraction preserves this source scale; physical-unit provenance
  must be resolved before physiological amplitude or power interpretation.
- Ruff passes and the full suite reports 212 passed and 2 skipped.

Pixel-wise Logistic Regression:

- The approved staged plan is stored in
  `.codex/memory-bank/plans/2026-06-14-logistic-regression-random-imagery.md`.
- The approved evaluation/CLI extension plan is stored in
  `.codex/memory-bank/plans/2026-06-15-logistic-regression-evaluation-cli.md`.
- Extension Stage 1, reconstruction metrics, is completed.
- Extension Stage 2, evaluation protocols and reusable runner, is completed.
- Extension Stage 3, evaluation artifacts and terminal CLI, is completed.
- Extension Stage 4, executable training notebook and final protocol comparison, is completed.
- The evaluation/CLI extension plan is completed.
- Stage 1, contracts, targets, grouped split, leakage checks, and non-EEG baselines, is completed.
- Stage 2, train-only common feature-family selection, is completed.
- Stage 3, per-pixel grouped grid search and one-time outer-test prediction, is completed.
- Stage 4, immutable experiment artifacts and validated pipeline round trips, is completed.
- Stage 5, final metrics, subject bootstrap uncertainty, figures, and the executed notebook, is
  implemented and awaiting explicit final review.
- Added reusable prediction metrics and subject-cluster bootstrap evaluation.
- Added and executed `notebooks/5.0-logistic-regression-random-pixels.ipynb` with five visually
  inspected figures and automated notebook validation.
- Final Logistic Regression mean per-pixel balanced accuracy is `0.509990919` with 95% subject
  bootstrap interval `[0.496383660, 0.521077288]`; the interval includes chance.
- Exact 6x6 reconstruction accuracy is zero. The pixel-frequency baseline has better bit accuracy,
  Brier score, and mean Hamming distance than Logistic Regression.
- Added foreground IoU at sample and global micro grain plus normalized Hamming loss for the model
  and every baseline. The reference model values are `0.335257970`, `0.334634146`, and
  `0.485754986`, respectively.
- Added typed cross-subject and identity-overlapping bidirectional cross-trial definitions,
  protocol-aware leakage audits, direction results, combined within-subject evaluation, and one
  reusable runner.
- Real protocol verification retains 141/39 rows for cross-subject evaluation. Within-subject
  evaluation includes 27 identities and 81/81 rows in each trial direction; subjects
  `14, 24, 27, 28, 29, 32` are excluded with provenance.
- The within-subject runner repeats feature screening and all per-pixel train-only decisions
  independently for each direction, accesses test features only after fitting, and combines
  predictions only after both directions complete.
- Added public train-or-reuse workflow shared by CLI and notebook.
- Published schema-v2 runs `4fcdf3c4fa5ef75a`, `ea7f8aa10a39cea0`, and
  `0ab4cb2a7512ab19`. The schema-v2 cross-subject run reproduces schema-v1 predictions exactly.
- Executed `notebooks/5.1-logistic-regression-training.ipynb` for both protocols and verified a
  second execution through immutable reuse.
- Cross-subject balanced accuracy is `0.509990919` with interval
  `[0.496383660, 0.521077288]`. Combined cross-trial balanced accuracy is `0.500013604` with
  interval `[0.486067242, 0.511482875]`; all protocol intervals include chance.
- Ruff passes and the full suite reports 263 passed; two existing Python 3.13 multiprocessing
  `fork()` deprecation warnings remain.
- The task uses only 180 `Data_Pattern/patt` random imagery blocks and 36 row-major binary pixel
  targets.
- The fixed outer protocol is subject-wise `GroupShuffleSplit(test_size=0.2, random_state=42)`;
  the expected real-corpus split is 141/39 rows from 26/7 disjoint subjects.
- Added strict target/split/baseline schemas and three non-EEG baselines: global majority,
  per-pixel frequency, and seeded Bernoulli.
- Real-corpus verification found no overlap in subjects, sample keys, seeds, or image payloads;
  every pixel has both classes in train and test.
- Added train-only canonical-key feature alignment, per-pixel five-fold stratified grouped CV,
  fold-local variance filtering, capped ANOVA selection, scaling, and fixed balanced L2 screening.
- Real screening selected `lbp` at mean per-pixel balanced accuracy 0.515542. `lgp` scored
  0.510085 and `time` 0.508035; all nine candidates were near chance, so the selection is
  procedural rather than evidence of held-out predictive value.
- A repeated screening run reproduced all candidate scores within `5e-10`; at the end of Stage 2,
  no feature manifests existed for the 39 outer-test rows. Stage 3 subsequently populated them
  only after all 36 train-only grid searches completed.
- Added 36 independent grouped `GridSearchCV` pipelines over the selected `lbp` family. Each grid
  covers 64 combinations of `k`, `C`, L1/L2, and class weight with every learned transform fitted
  inside folds.
- The real run produced finite `(39, 36)` probabilities and predictions. Mean best CV balanced
  accuracy was 0.579192, while mean outer-test per-pixel balanced accuracy was 0.509991
  (range 0.360963-0.658730), indicating selection optimism and weak subject generalization.
- Added atomic immutable experiment runs with complete config/environment/split/screening/grid
  metadata, SHA-256 file inventory, 36 joblib pipelines, stored targets/predictions, and an
  explicit trusted-load boundary.
- Published and validated
  `artifacts/experiments/logistic-regression/f515948b6bf5af55/` (about 14 MiB, 48 tracked payload
  files plus manifest). Pipeline replay reproduces stored probabilities bit-for-bit.
- The run records commit `1ca50bf23fdbffb79609a80bacb2f7884e4ac8bc` with `git_dirty=true`.
- Ruff passes and the full suite reports 251 passed.

Classical random-imagery models:

- The approved staged plan is stored in
  `.codex/memory-bank/plans/2026-06-15-classical-random-imagery-models.md`.
- Stage 1, common model-agnostic experiment framework, is completed.
- Stage 2, calibrated Linear SVM and Ridge Classifier backends, is completed.
- Stage 3, independent and multi-output regression backends, is completed.
- Stage 4, schema-v3 artifacts, workflows, and CLI, is completed.
- Stage 5, full real-corpus training for all nine variants and both protocols, is completed.
- Stage 6, final protocol-separated model comparison, is implemented and awaiting explicit
  review.
- Planned variants cover calibrated Linear SVM and Ridge Classifier plus independent and
  multi-output Ridge, ElasticNet, Random Forest, and multi-output PLS.
- Existing Logistic Regression APIs, CLI behavior, schema-v1/v2 readers, and reference predictions
  are compatibility constraints for the common framework.
- Added `experiments/random_imagery` with the model registry, shared configuration, bounded-score
  contracts, backend-driven protocol runner, and Logistic Regression adapter.
- Added typed default configurations and independent backends for Linear SVM and Ridge Classifier.
- Each classifier performs model-specific train-only feature-family screening and per-pixel grouped
  grid search, then fits a scalar Logistic Regression Platt calibrator from grouped OOF decision
  scores only.
- Final classifier pipelines are trained on all direction-training rows; outer-test features are
  materialized only after all 36 pixel pipelines and calibrators are complete.
- Synthetic cross-subject and bidirectional cross-trial runs produce deterministic finite
  float64 scores in `[0, 1]` with exact threshold-derived labels.
- Added independent and multi-output Ridge, ElasticNet, and Random Forest backends plus
  exploratory multi-output PLS.
- Independent regressors retain one fitted pipeline per pixel. Multi-output regressors retain one
  shared selector, pipeline, and hyperparameter set for every target.
- Regression feature screening and search maximize thresholded balanced accuracy, break ties with
  lower clipped validation MSE, and then preserve configured candidate order.
- Added fold-local multi-target `f_classif` percentile ranking with deterministic original-index
  tie resolution.
- Raw regressor outputs are clipped to `[0, 1]` only after lower/upper clipping fractions are
  measured. Scores remain continuous outputs and are not treated as probabilities.
- Added schema-v3 atomic immutable artifacts under model-specific roots. Run identity includes the
  model configuration, protocol, and direction; manifests inventory every payload with SHA-256 and
  byte size.
- Safe schema-v3 evaluation validates all metadata, arrays, metrics, bootstrap summaries, and
  pipeline paths without loading joblib. Trusted replay additionally validates feature identity
  and exact test sample keys before deserializing the manifested pipelines.
- Independent variants persist one pipeline per target; multi-output variants persist one shared
  pipeline. Calibrated classifier artifacts persist Platt coefficients separately from the base
  pipelines.
- Added shared train-or-reuse orchestration and equivalent `random-imagery-models` and
  `python -m experiments.random_imagery` entry points with `run`, `evaluate`, and `compare`.
- Mixed schema-v2 Logistic Regression and schema-v3 model comparison is supported only when
  protocol, direction, and ordered test sample keys are identical.
- Executed `notebooks/5.2-classical-models-training.ipynb` top-to-bottom and then re-executed all
  18 model/protocol workflow calls through immutable reuse.
- Published exactly 27 active schema-v3 direction runs: nine cross-subject runs with 141/39 rows
  and eighteen cross-trial runs with 81/81 rows.
- Safe validation passed for every run. All leakage audits are clean, every target retains both
  classes, independent variants contain 36 pipelines, and multi-output variants contain one.
- Real-corpus convergence requires independent ElasticNet `max_iter=1_000_000`, `tol=1e-4` and
  multi-output ElasticNet `max_iter=1_000_000`, `tol=1e-3`; convergence warnings remain errors.
- The current schema-v3 model artifact set occupies about 127 MiB.
- Added `experiments/random_imagery/comparison.py` for exact schema-v2/v3 alignment, shared
  subject-cluster bootstrap draws, paired metric improvements, pooled classifier calibration,
  and regressor clipping summaries.
- Executed and visually inspected `notebooks/5.3-classical-models-comparison.ipynb`; all four
  figures and marker `CLASSICAL_MODELS_COMPARISON_VERIFIED` passed.
- No model has a pointwise paired 95% balanced-accuracy improvement interval excluding zero versus
  Logistic Regression in either protocol.
- Cross-subject descriptive leaders are independent Ridge Regression (`0.518381515`) and
  independent Random Forest (`0.518205956`). Combined within-subject descriptive leader is
  multi-output ElasticNet/Lasso (`0.503456996`).
- Grouped Platt calibration lowers pooled ECE from Logistic Regression's `0.248079` to `0.052025`
  for Linear SVM and `0.032203` for Ridge Classifier cross-subject, and from `0.279373` to
  `0.064972` and `0.056217` within-subject.
- The global-majority baseline has lower score-MSE than every learned model in both protocols.
  Exact 36-pixel reconstruction remains zero for every model.
- Full verification reports 334 passed with two pre-existing Python 3.13 multiprocessing warnings;
  Ruff, lockfile, notebook integration, visual, and diff checks pass.
- A real target-only audit validated five shared subject-grouped folds across all 180 rows and 36
  pixels; every train and validation fold retained both target classes.
- Canonical target/protocol, baseline, and metric implementations now live in the common package;
  existing Logistic Regression module paths remain compatibility wrappers.
- Common-runner parity is exact for synthetic cross-subject and cross-trial protocols. Trusted
  replay of reference run `f515948b6bf5af55` reproduces its `(39, 36)` probabilities and labels
  bit-for-bit.
- Ruff passes and the full suite reports 330 passed; two existing Python 3.13 multiprocessing
  `fork()` deprecation warnings remain.

## Next Actions

- Obtain explicit final approval for Stage 6.
- The original Logistic Regression Stage 5 remains awaiting separate review; the extension's
  reconstruction metrics are already incorporated.
- Obtain explicit final approval for EEG feature-extraction Stage 5 separately.
- Benchmark full-corpus cache warmup only when operational timing is needed.

## Open Questions

- No unresolved questions remain for the completed evaluation/CLI extension.
