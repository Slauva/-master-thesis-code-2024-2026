# Experiments

## 2026-06-16 - Torch/classical random-imagery final comparison

- Artifact: executed `notebooks/6.1-torch-classical-comparison.ipynb`.
- Scope: Logistic Regression reference, nine classical schema-v3 models, 12 Torch spectral models,
  and canonical non-EEG baselines.
- Loading boundary: compared only immutable metadata and arrays. The notebook uses
  `load_evaluation_run`, `load_model_run`, and `load_torch_run`; it does not deserialize joblib
  pipelines or Torch checkpoint weights.
- Protocol alignment: every non-reference model is required to match Logistic Regression exactly
  on ordered test sample keys, target matrices, and subject IDs. Cross-subject uses 39 held-out
  rows from seven subjects. Combined bidirectional cross-trial uses 162 held-out rows from
  27 identities.
- Bootstrap contract: 2,000 accepted subject-cluster bootstrap draws are shared by every model
  within a protocol. Cross-subject required 2,002 draw attempts; within-subject required 2,000.
- Multiplicity: balanced-accuracy bootstrap p-values are Holm-adjusted across the 21
  non-reference learned models in each protocol. The minimum adjusted p-value is `0.273000`, so no
  model is promoted as superior to Logistic Regression.
- Descriptive cross-subject leader: `ridge-regression-independent`, balanced accuracy `0.518382`.
  The top Torch cross-subject variant is `shallow-convnet-morlet-multilabel`, balanced accuracy
  `0.513443`.
- Descriptive combined within-subject leader: `deep-convnet-stft-multilabel`, balanced accuracy
  `0.512011`.
- Score semantics: Logistic Regression, calibrated Linear SVM/Ridge Classifier, and Torch models
  are probability-score models and receive pooled fixed-bin ECE. Classical regressors expose
  clipped continuous scores and are explicitly excluded from calibration interpretation.
- Exact 36-pixel reconstruction accuracy remains zero for every learned model in the final
  comparison.
- Visual QA: six figures inspected: balanced-accuracy rankings by protocol, paired
  balanced-accuracy improvements by protocol, probability-model ECE, and Torch runtime versus
  parameter count.
- Verification: focused Stage 6 checks reported 10 passed; `uv run ruff check .`,
  `uv lock --check`, and `git diff --check` passed; full `uv run pytest` reported 430 passed with
  two pre-existing Python 3.13 multiprocessing `fork()` deprecation warnings.
- Interpretation boundary: the comparison supports a conservative thesis conclusion that current
  learned EEG models remain near chance on this random-imagery reconstruction task. Descriptive
  ranks are not evidence of reliable superiority.

## 2026-06-16 - Torch spectral random-imagery real-corpus training

- Artifact: executed `notebooks/6.0-torch-spectral-models-training.ipynb`.
- Scope: 12 primary Torch variants, crossing EEGNet, DeepConvNet, and ShallowConvNet with FFT,
  Morlet, Superlet, and STFT crop-spectral imagery inputs.
- Protocols: one cross-subject direction with 141 train rows and 39 held-out rows; two
  identity-overlapping cross-trial directions with 81 train and 81 test rows each.
- Training contract: one 36-logit multilabel model per variant/direction, final seeds 42, 43, and
  44, train-only log-power frequency z-scoring, train-only positive weights, grouped validation
  epoch selection, and fixed threshold 0.5.
- Artifacts: 36 immutable Torch direction runs under
  `artifacts/experiments/random-imagery-torch/`, each with three state-dict checkpoints,
  histories, normalization state, split/leakage audit, ensemble scores, predictions, metrics,
  baselines, environment payload, and SHA-256 file inventory.
- Reuse check: the notebook's second pass called the same workflow with `reuse_existing=True` and
  verified reuse for all 36 expected direction runs without fitting.
- Runtime/environment: artifact environment payloads record CUDA available on
  `NVIDIA GeForce RTX 3070 Ti`; summed persisted training time across direction runs was about
  1,231.64 s.
- Direction metrics: cross-subject mean per-pixel balanced accuracy ranged from `0.486743` to
  `0.513443` with mean `0.500453`; within-subject direction balanced accuracy ranged from
  `0.479567` to `0.524497` with mean `0.502307`.
- Combined within-subject descriptive leader: `deep-convnet-stft-multilabel`, balanced accuracy
  `0.512011`, 95% subject-bootstrap interval `[0.500668, 0.520872]`, 162 test rows.
- Parameter counts across persisted Torch variants ranged from 2,495 to 184,640 trainable
  parameters.
- Visual QA: two notebook figures were inspected, one per protocol. Cross-subject has one
  direction per variant; within-subject shows both trial directions per variant.
- Verification: 97 focused Torch/notebook tests passed; `uv run ruff check .`, `uv lock --check`,
  and `git diff --check` passed; full `uv run pytest` reported 429 passed with two pre-existing
  Python 3.13 multiprocessing `fork()` deprecation warnings.
- Interpretation boundary: Stage 5 validates completion, leakage-safe execution, artifact safety,
  and descriptive metrics only. Multiplicity-aware contrasts against Logistic Regression and
  classical models are deferred to Stage 6.

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

## 2026-06-14 - Random imagery target and split audit

- Scope: Stage 1 contract audit for the pixel-wise Logistic Regression baseline; no EEG features
  or fitted classifiers were used.
- Source: all 180 `type="random"` blocks from `Data_Pattern/patt`.
- Targets: row-major binary `(180, 36)` matrix with names `pixel_r0_c0` through `pixel_r5_c5`.
- Split: `GroupShuffleSplit(test_size=0.2, random_state=42)` grouped by subject, producing 141
  train rows from 26 subjects and 39 test rows from subjects 9, 10, 16, 18, 20, 28, and 33.
- Class support: train positive counts per pixel range 59-81; test positive counts range 14-26;
  every pixel contains both classes in both partitions.
- Leakage audit: no shared subjects, canonical sample keys, seeds, or SHA-256 image fingerprints.
- Baselines prepared for later evaluation: global majority, per-pixel training frequency, and
  seeded Bernoulli; each produces `(39, 36)` probabilities and predictions.
- Verification: 24 focused tests passed, Ruff passed, `git diff --check` passed, and the full suite
  reported 236 passed and 2 skipped.

## 2026-06-15 - Random imagery train-only feature-family screening

- Scope: Stage 2 common-family selection for 36 binary pixel tasks; the 39 outer-test rows remained
  untouched.
- Inputs: 141 full-epoch `[0.5, 15.5)` `Data_Pattern/patt` random-imagery blocks from 26
  outer-train subjects. Each block produced one feature row.
- Cross-validation: per-pixel five-fold `StratifiedGroupKFold(shuffle=True, random_state=42)` by
  subject. All 180 pixel-fold combinations had both classes in train and validation; validation
  folds contained 24-33 rows from 4-7 subjects.
- Fold-local pipeline: variance threshold 0, capped ANOVA `SelectKBest(k=100)`, standardization,
  and balanced L2 Logistic Regression with `C=1`, `liblinear`, and `max_iter=5000`.
- Mean per-pixel balanced accuracy: `time` 0.508035, `spectral` 0.501098,
  `time+spectral` 0.505047, `covariance` 0.494298, `correlation` 0.490803,
  `log_covariance` 0.500776, `lndp` 0.501745, `lgp` 0.510085, and `lbp` 0.515542.
- Selection: `lbp` by the predefined maximum-mean rule. Its matrix shape was `(141, 16128)`;
  every fold retained 100 selected features.
- Reproducibility: a second complete cached run reproduced every score within `5e-10`.
- Leakage check: feature manifests remained absent for all 39 outer-test keys after both runs.
- Interpretation boundary: all candidate means are near chance and the leading margin is small.
  This stage chooses a common train-only representation; it does not estimate held-out performance.
- Verification: 29 experiment tests passed without warnings, Ruff passed, `git diff --check`
  passed, and the full suite reported 243 passed with two pre-existing multiprocessing warnings.

## 2026-06-15 - Random imagery per-pixel Logistic Regression grid search

- Scope: Stage 3 training and one outer-test prediction pass for 36 independent binary pixel
  models using the Stage 2-selected `lbp` family.
- Inputs: 141 outer-train rows from 26 subjects and 39 outer-test rows from 7 disjoint subjects;
  one full `[0.5, 15.5)` epoch per row and 16,128 original `lbp` columns.
- Search: one five-fold subject-grouped `GridSearchCV` per pixel, balanced accuracy, fixed Stage 2
  folds, and 64 combinations from `k={25,50,100,250}`, `C={0.01,0.1,1,10}`, L1/L2, and
  class weight `{None,balanced}`.
- Pipeline: variance threshold 0, capped ANOVA selection, standardization, and `liblinear`
  Logistic Regression with `max_iter=5000`. All transforms were fitted inside folds; final refit
  used all 141 training rows.
- Execution: `n_jobs=-1`; all 36 searches plus test prediction completed in 43.958 s. Temporary
  joblib transform caches were deleted and removed from returned pipelines.
- Outputs: finite `(39,36)` probabilities and binary predictions, finite selected coefficients,
  and 64 retained candidate CV summaries per pixel.
- Best-CV balanced accuracy: mean 0.579192, range 0.500000-0.684022.
- Outer-test per-pixel balanced accuracy: mean 0.509991, standard deviation 0.085920, range
  0.360963-0.658730.
- Hyperparameters: `k` counts 25/50/100/250 = 6/9/8/13; `C` counts
  0.01/0.1/1/10 = 6/11/12/7; L1/L2 = 17/19; class weight None/balanced = 22/14.
- Optimizer iterations ranged from 0 to 13. A zero-iteration final fit is valid for liblinear when
  the initial solution already meets its stopping condition.
- Interpretation: the mean selected CV score exceeds the mean held-out score by about 0.0692,
  while mean held-out balanced accuracy is near chance. This is consistent with hyperparameter
  selection optimism and weak cross-subject signal, not successful 6x6 reconstruction.
- Verification: exact deterministic predictions and choices passed on synthetic grouped data;
  a real train-only rerun reproduced pixel 0's choice and CV score; Ruff passed and the full suite
  reported 246 passed with two pre-existing multiprocessing warnings.

## 2026-06-15 - Random imagery immutable experiment artifact

- Artifact:
  `artifacts/experiments/logistic-regression/f515948b6bf5af55/`.
- Materialization: reproduced the Stage 2 screening and Stage 3 36-pixel grid search from feature
  cache, then published the complete run through a hidden sibling directory and atomic rename.
- Contents: resolved config, environment, outer split, 180 grouped CV folds, 16,128 feature names,
  all screening results, all 2,304 grid candidate summaries, selected feature supports and
  coefficients, train/test targets, probabilities, predictions, per-pixel test scores, and 36
  fitted pipelines.
- Integrity: 48 payload files are listed with exact sizes and SHA-256 in `manifest.json`; total
  directory size is about 14 MiB and total file count is 49 including the manifest.
- Trust boundary: joblib loading is denied by default and requires explicit `trusted=True` after
  all manifest checks. Pipeline filenames are constrained to manifested local files under
  `pipelines/`.
- Replay: all 36 pipelines loaded successfully and reproduced the stored `(39,36)` probabilities
  bit-for-bit and labels exactly from canonical aligned test features.
- Result consistency: persisted mean outer-test per-pixel balanced accuracy is 0.509990919, equal
  to Stage 3.
- Environment: Python 3.13.11, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.9.0, joblib 1.5.3,
  MNE 1.11.0, Pydantic 2.12.5, and OmegaConf 2.3.0.
- Git provenance: commit `1ca50bf23fdbffb79609a80bacb2f7884e4ac8bc`, dirty working tree.
  Artifact payload integrity is validated, but final thesis provenance should use a committed
  implementation revision.
- Verification: five focused artifact tests covered trusted round trip, prediction replay,
  duplicate refusal, missing/corrupt files, and unsafe pipeline paths. Ruff passed,
  `git diff --check` passed, and the full suite reported 251 passed with two pre-existing
  multiprocessing warnings.

## 2026-06-15 - Random imagery final Logistic Regression evaluation

- Artifact: executed `notebooks/5.0-logistic-regression-random-pixels.ipynb`, reading immutable
  run `artifacts/experiments/logistic-regression/f515948b6bf5af55/` without retraining.
- Evaluation set: 39 random-imagery blocks from 7 held-out subjects; one 36-pixel target per block.
- Primary result: mean per-pixel balanced accuracy `0.509990919`.
- Uncertainty: percentile cluster bootstrap over complete subjects, 2,000 valid resamples,
  `random_state=42`; 95% interval `[0.496383660, 0.521077288]`. Two additional draws were rejected
  because at least one pixel lost a target class.
- Other model metrics: mean macro F1 `0.499652645`, bit accuracy `0.514245014`, mean Brier score
  `0.333966692`, exact-match accuracy `0`, and mean Hamming distance `17.487179` of 36 pixels.
- Non-EEG comparison: global-majority and pixel-frequency balanced accuracy were both `0.5`;
  seeded Bernoulli was `0.500395206`. Pixel frequency outperformed Logistic Regression on bit
  accuracy (`0.524216524`), Brier score (`0.250529325`), and Hamming distance (`17.128205`).
- Selection optimism: mean selected inner-CV balanced accuracy `0.579192` exceeded held-out
  performance by about `0.0692`.
- Visual evidence: five inspected figures cover uncertainty and baselines, train-only feature
  screening plus CV/test scores, the 6x6 per-pixel score map, deterministic closest/median/farthest
  reconstructions, and descriptive LBP channel selection counts.
- Interpretation: the bootstrap interval includes chance, no method exactly reconstructed any 6x6
  image, and the EEG model does not dominate simple frequency baselines. This is a reproducible
  negative cross-subject baseline, not evidence of successful image reconstruction.
- Provenance caveats: only seven test subjects; source physical-unit scale remains unresolved; the
  persisted run records a dirty Git worktree and should be tied to a committed revision for thesis
  reporting.
- Verification: 9 notebook code cells executed with no errors and 5 stored PNG outputs; 43 focused
  experiment/notebook tests passed; Ruff and `git diff --check` passed; full suite reported
  257 passed with two pre-existing multiprocessing warnings.

## 2026-06-15 - Random imagery reconstruction metric extension

- Scope: evaluation-only extension of the immutable cross-subject run
  `artifacts/experiments/logistic-regression/f515948b6bf5af55/`; no fitting, feature selection,
  threshold selection, or artifact rewrite occurred.
- Logistic Regression: mean sample foreground IoU `0.335257970`, global micro foreground IoU
  `0.334634146`, and normalized Hamming loss `0.485754986`.
- Global-majority baseline: mean sample IoU `0`, micro IoU `0`, Hamming loss `0.478632479`.
- Pixel-frequency baseline: mean sample IoU `0.293694463`, micro IoU `0.291622481`, Hamming loss
  `0.475783476`.
- Seeded-Bernoulli baseline: mean sample IoU `0.326008812`, micro IoU `0.324298161`, Hamming loss
  `0.497150997`.
- Invariants: empty-target/empty-prediction sample IoU is `1.0`;
  `hamming_loss = 1 - bit_accuracy = mean_hamming_distance / 36`.
- Notebook: `notebooks/5.0-logistic-regression-random-pixels.ipynb` re-executed with all 9 code
  cells successful, no error outputs, the integration marker present, and 5 figures visually
  inspected.
- Verification: focused metric/notebook tests reported 8 passed; Ruff and `git diff --check`
  passed; full suite reported 259 passed with two pre-existing multiprocessing warnings.

## 2026-06-15 - Logistic Regression evaluation protocol and runner validation

- Scope: Stage 2 protocol, leakage-audit, and orchestration validation. No real model fitting or
  new immutable experiment artifact was produced.
- Cross-subject protocol: retained the fixed 141 train rows from 26 subjects and 39 test rows from
  7 disjoint subjects.
- Bidirectional cross-trial eligibility: 27 subjects with both trials; eligible IDs are
  `1-13, 15-21, 23, 25, 26, 30, 31, 33, 34`. Excluded IDs are
  `14, 24, 27, 28, 29, 32`.
- Trial 1 -> Trial 2: 81 train and 81 test rows. Per-pixel positive counts range 31-51 in train
  and 28-47 in test.
- Trial 2 -> Trial 1: 81 train and 81 test rows. Per-pixel positive counts range 28-47 in train
  and 31-51 in test.
- Leakage audits: the expected 27 subject identities overlap in each cross-trial direction; sample
  keys, trial numbers, random seeds, and image fingerprints do not overlap. Every one of the 36
  pixel tasks has both classes in train and test.
- Synthetic execution: one runner completed cross-subject and both cross-trial directions with
  deterministic probabilities and predictions. Event-order tests showed screening data, screening,
  and grid fitting before test-feature access in each direction; combination occurred only after
  both predictions.
- Combined clustering check: synthetic subjects contributed six test rows each after combining
  both directions, matching the real protocol's intended subject-bootstrap grain.
- Verification: 48 experiment tests passed; Ruff and `git diff --check` passed; full suite
  reported 263 passed with two pre-existing multiprocessing warnings.

## 2026-06-15 - Logistic Regression evaluation artifact and CLI validation

- Scope: Stage 3 structural and synthetic validation. No real schema-v2 model run or notebook was
  produced.
- Schema-v1 compatibility: safely evaluated immutable run
  `artifacts/experiments/logistic-regression/f515948b6bf5af55/` without joblib. Recomputed model
  balanced accuracy `0.509990919`, mean sample IoU `0.335257970`, micro IoU `0.334634146`, and
  Hamming loss `0.485754986`.
- Schema-v2 synthetic coverage: wrote and loaded protocol-aware cross-subject and both
  within-subject directions with exact inventory validation, `evaluation.json`, baseline arrays,
  duplicate refusal, corruption rejection, and metric/bootstrap consistency checks.
- Combined within-subject validation: loaded the two immutable directions in reverse input order,
  restored canonical direction order, combined 72 synthetic test rows from 12 subjects, and
  reproduced the runner's combined mean balanced accuracy without modifying either manifest.
- CLI coverage: parsed repeatable dotted OmegaConf overrides; exercised new training, duplicate
  refusal, complete-set reuse without fitting, schema-v1 JSON evaluation, failure exit codes, and
  equivalent console-script/module help output.
- Packaging: added an editable setuptools project entry so `uv run logistic-regression` installs
  alongside the existing `uv run python -m experiments.logistic_regression` form.
- Verification: 62 experiment tests passed; Ruff, lockfile check, and `git diff --check` passed;
  full suite reported 277 passed with two pre-existing Python 3.13 multiprocessing warnings.

## 2026-06-15 - Logistic Regression protocol training and final comparison

- Notebook: executed `notebooks/5.1-logistic-regression-training.ipynb`, then executed it again
  with `REUSE_EXISTING=True`. All 8 code cells completed, no error outputs were stored, one
  uncertainty figure was visually inspected, and
  `LOGISTIC_REGRESSION_TRAINING_PROTOCOLS_VERIFIED` is present.
- Cross-subject schema-v2 artifact: `artifacts/experiments/logistic-regression/4fcdf3c4fa5ef75a/`.
  It contains 55 inventoried payload files and reproduces schema-v1 run `f515948b6bf5af55`
  probabilities, predictions, and test targets exactly.
- Cross-subject result: selected `lbp`; mean best inner-CV balanced accuracy `0.579192334`;
  held-out balanced accuracy `0.509990919`, 95% complete-subject bootstrap interval
  `[0.496383660, 0.521077288]`, mean sample IoU `0.335257970`, micro IoU `0.334634146`, and
  Hamming loss `0.485754986`.
- Trial 1 -> Trial 2 artifact:
  `artifacts/experiments/logistic-regression/ea7f8aa10a39cea0/`. It selected `correlation`;
  mean best inner-CV balanced accuracy `0.581309200`; held-out balanced accuracy `0.503108711`,
  interval `[0.484477246, 0.521643876]`, mean sample IoU `0.334078093`, and Hamming loss
  `0.494513032`.
- Trial 2 -> Trial 1 artifact:
  `artifacts/experiments/logistic-regression/0ab4cb2a7512ab19/`. It selected `time+spectral`;
  mean best inner-CV balanced accuracy `0.601988274`; held-out balanced accuracy `0.499795634`,
  interval `[0.480587839, 0.518659408]`, mean sample IoU `0.337628447`, and Hamming loss
  `0.498285322`.
- Combined bidirectional cross-trial evaluation: 162 held-out rows from 27 eligible identities;
  balanced accuracy `0.500013604`, interval `[0.486067242, 0.511482875]`, mean sample IoU
  `0.335853270`, micro IoU `0.335094166`, and Hamming loss `0.496399177`.
- Interpretation: every protocol and direction interval includes chance. Identity overlap did not
  improve held-out reconstruction under this cross-trial protocol. Inner-CV scores remain higher
  than held-out scores, so selection optimism persists. The protocol difference is not a pure
  model effect because populations and transfer targets differ.
- Verification: focused notebook/workflow tests reported 11 passed; Ruff, lockfile, and diff
  checks passed; full suite reported 280 passed with two pre-existing Python 3.13 multiprocessing
  warnings.

## 2026-06-15 - Common random-imagery framework compatibility validation

- Scope: structural Stage 1 validation only; no Linear SVM, Ridge Classifier, regression model, or
  new real experiment artifact was trained.
- Registry: one Logistic Regression reference plus nine planned model variants with explicit
  task, topology, score semantics, and exploratory/reference metadata.
- Synthetic parity: the common Logistic Regression backend reproduced legacy selected feature
  families, scores, labels, metrics, and subject-bootstrap samples exactly under cross-subject
  and bidirectional cross-trial protocols.
- Test-access ordering: the common runner completed backend fitting before materializing
  outer-test features and predicting.
- Real compatibility: safe CLI evaluation passed for schema-v1 run `f515948b6bf5af55` and
  schema-v2 run `4fcdf3c4fa5ef75a`. Trusted schema-v1 pipeline replay reproduced stored
  probabilities and predictions bit-for-bit with shape `(39, 36)`.
- Verification: 70 experiment tests passed; Ruff, lockfile, and diff checks passed; the full suite
  reported 286 passed with two pre-existing Python 3.13 multiprocessing warnings.

## 2026-06-15 - Grouped OOF calibration validation

- Scope: Stage 2 structural and synthetic validation for independent Linear SVM and Ridge
  Classifier backends. No real-corpus model or immutable schema-v3 artifact was produced.
- Feature selection: each classifier screened candidate feature families with its own estimator on
  pixel-specific grouped folds, then ran an independent per-pixel grouped hyperparameter search.
- Calibration: the selected pipeline was cloned and refitted per grouped fold; every outer-train
  row received exactly one OOF decision score from a model that excluded its validation group.
  A scalar Logistic Regression calibrator mapped these OOF scores into `[0, 1]`.
- Final fitting: one selected base pipeline per pixel was refitted on all direction-training rows.
  Test features were materialized only after every pixel pipeline and calibrator had completed.
- Synthetic cross-subject validation: both models produced deterministic finite float64 score
  matrices, exact `0.5` threshold-derived int8 predictions, complete OOF fold assignments, and
  disjoint train/validation subjects in every calibration fold.
- Synthetic bidirectional cross-trial smoke validation: both models completed Trial 1 -> Trial 2
  and Trial 2 -> Trial 1 and combined the two independently calibrated predictions successfully.
- Verification: 76 experiment tests passed; Ruff, lockfile, and diff checks passed; the full suite
  reported 292 passed with two pre-existing Python 3.13 multiprocessing warnings.

## 2026-06-15 - Classical regression backend validation

- Scope: Stage 3 structural, synthetic, and target-only validation for seven regression variants.
  No real EEG feature training or immutable schema-v3 artifact was produced.
- Variants: independent and multi-output Ridge, ElasticNet, and Random Forest regression plus
  exploratory multi-output PLS.
- Independent topology: one pixel-specific grouped search and final fitted pipeline per target.
  Multi-output topology: one shared subject-grouped fold partition, selector, search, pipeline,
  and hyperparameter set across all targets.
- Selection: model-specific feature-family screening and grid search maximize thresholded balanced
  accuracy; exact ties use lower clipped validation MSE and then configured order. Screening
  estimator parameters are explicit rather than inherited from the first search-grid candidate.
- Multi-target selector: fold-local `f_classif` percentile ranks are averaged across targets with
  deterministic original-feature-index tie resolution.
- Scores: raw predictions are checked for finiteness, lower and upper clipping fractions are
  measured, and outputs are clipped into `[0, 1]` before configured-threshold labels are formed.
- Synthetic validation: all seven variants produced deterministic bounded scores. Independent and
  multi-output Ridge also produced exact `(18, 36)` test matrices; representative Ridge and PLS
  variants completed both cross-trial directions and combined successfully.
- Real target audit: all 180 random-imagery rows and 36 targets from 33 subjects passed shuffled
  five-fold shared `GroupKFold` validation. Train/validation row counts were `141/39`, `141/39`,
  `141/39`, `153/27`, and `144/36`; every pixel retained both classes in every partition.
- Verification: 98 experiment tests passed; Ruff, lockfile, and diff checks passed; the full suite
  reported 316 passed with two pre-existing Python 3.13 multiprocessing warnings.

## 2026-06-15 - Schema-v3 model artifact and CLI validation

- Scope: Stage 4 structural and synthetic validation. No real-corpus model run or training
  notebook was produced.
- Artifacts: added model-specific schema-v3 run roots, deterministic configuration hashes, atomic
  publication, complete SHA-256/size inventories, persisted train/test targets, scores,
  predictions, baselines, diagnostics, and topology-aware pipelines.
- Safe boundary: metadata/array evaluation recomputes and validates persisted metrics and
  bootstrap summaries without calling `joblib.load`. Pipeline counts and manifested relative
  paths are validated even in safe mode; corruption, extra files, and unsafe `..` paths are
  rejected.
- Trusted replay: independent Linear SVM, independent Ridge Regression, and multi-output PLS
  synthetic runs reproduced persisted scores to numerical precision and labels exactly after
  validating feature blocks, ordered feature names, and canonical test sample keys.
- Workflow: one public function trains or reuses a complete immutable protocol run set. Synthetic
  within-subject reuse loaded both directions without refitting and recomputed the combined
  evaluation without rewriting manifests; incomplete sets were rejected.
- CLI: added equivalent installed and module entry points with `run`, safe `evaluate`, and
  compatible `compare` commands. A mixed schema-v2 Logistic Regression/schema-v3 PLS comparison
  succeeded on an identical synthetic cross-subject split.
- Verification: 113 experiment tests passed; Ruff, lockfile, and diff checks passed; the full
  suite reported 329 passed with two pre-existing Python 3.13 multiprocessing warnings.

## 2026-06-15 - Full real-corpus classical-model training

- Notebook: executed `notebooks/5.2-classical-models-training.ipynb` top-to-bottom, then validated
  a complete second pass through immutable reuse. Seven code cells are executed, no error output
  is stored, one descriptive run-level figure was visually inspected, and
  `CLASSICAL_MODELS_TRAINING_VERIFIED` is present.
- Corpus and protocols: all models use the same 180 `Data_Pattern/patt`, `type="random"` rows,
  33 subjects, 36 row-major targets, and feature config hash `fb8c5dcc8a1d3f30`. Cross-subject
  directions use 141/39 rows; both cross-trial directions use 81/81 rows from 27 eligible
  identities.
- Artifact set: exactly 27 active schema-v3 direction runs occupy about 127 MiB. Every safe load
  passed complete inventory, metric, bootstrap, split, leakage, and topology validation.
- Numerical adjustment: independent ElasticNet uses `max_iter=1_000_000`, `tol=1e-4`.
  Multi-output ElasticNet uses `max_iter=1_000_000`, `tol=1e-3` after strict real-corpus runs
  showed that `tol=1e-4` remained unmet on one grouped fold. Convergence warnings are still
  promoted to errors.
- Linear SVM: cross-subject `f3623d36e070c677`, LBP, balanced accuracy `0.493335880`;
  Trial 1 -> 2 `72852afb69efda8d`, LBP, `0.503029993`; Trial 2 -> 1
  `8e9ae1ac1fd4756d`, time+spectral, `0.512548464`.
- Ridge Classifier: cross-subject `6343dec0600355e7`, LGP, `0.502153502`; Trial 1 -> 2
  `b87938f8647ad532`, log-covariance, `0.503065128`; Trial 2 -> 1
  `3b9e5e1624098afb`, time+spectral, `0.513105140`.
- Independent Ridge Regression: cross-subject `c7605762c2e4c898`, LGP, `0.518381515`;
  Trial 1 -> 2 `4eb9fbae64345f4b`, log-covariance, `0.501807861`; Trial 2 -> 1
  `20630da85752c082`, time+spectral, `0.496847788`.
- Multi-output Ridge Regression: cross-subject `0af85371b36da04f`, time+spectral,
  `0.497668271`; Trial 1 -> 2 `643f046566cabab0`, time, `0.510087502`; Trial 2 -> 1
  `993e9bc2c622349b`, spectral, `0.499272610`.
- Independent ElasticNet: cross-subject `1ebfde4e234d58ff`, LBP, `0.512398951`;
  Trial 1 -> 2 `4e256d04ea981fef`, spectral, `0.495357334`; Trial 2 -> 1
  `85d203a041f1db98`, LBP, `0.510503974`.
- Multi-output ElasticNet: cross-subject `b043eff18b45f217`, time+spectral, `0.506086304`;
  Trial 1 -> 2 `82ad1db8178806a7`, spectral, `0.506051733`; Trial 2 -> 1
  `c473d0e79c57fa14`, time+spectral, `0.503266468`.
- Independent Random Forest: cross-subject `e6f27a2ba93abb13`, LBP, `0.518205956`;
  Trial 1 -> 2 `92a708813a79889f`, spectral, `0.506584690`; Trial 2 -> 1
  `db54f50d9cb2cc05`, time+spectral, `0.504821687`.
- Multi-output Random Forest: cross-subject `35ee5634a4c5e481`, time, `0.504499179`;
  Trial 1 -> 2 `88cdaaf9fbd21cf4`, spectral, `0.506629297`; Trial 2 -> 1
  `518ad5e5e02cfd19`, LNDP, `0.508794036`.
- Multi-output PLS: cross-subject `b9d0ebd252d5d2d9`, correlation, `0.498280588`;
  Trial 1 -> 2 `4fa8edfdd04111e1`, spectral, `0.502811732`; Trial 2 -> 1
  `351a10357b7e11d5`, correlation, `0.509864597`.
- These direction-level values are descriptive training validation only. Cross-subject and
  cross-trial estimates are not averaged, and isolated confidence intervals above chance are not
  interpreted without the planned paired, multiple-model comparison in Stage 6.
- Verification: notebook integration and visual checks passed; Ruff, lockfile, and diff checks
  passed; the full suite reported 330 passed with two pre-existing Python 3.13 multiprocessing
  warnings.

## 2026-06-15 - Final classical-model comparison

- Notebook: executed `notebooks/5.3-classical-models-comparison.ipynb` with ten code cells, no
  error output, four visually inspected figures, and marker
  `CLASSICAL_MODELS_COMPARISON_VERIFIED`.
- Compatibility: every schema-v3 candidate exactly matched the schema-v2 Logistic Regression
  reference on ordered test sample keys, binary targets, and subject IDs. Cross-subject retained
  39 rows from seven subjects; combined bidirectional cross-trial retained 162 rows from 27
  subjects.
- Uncertainty: each protocol used 2,000 accepted subject-cluster bootstrap draws with seed 42,
  shared across all models and paired reference differences. Cross-subject required 2,002 draw
  attempts because two resamples lacked both classes for at least one pixel.
- Cross-subject balanced accuracy ranged from `0.493335880` to `0.518381515`. Independent Ridge
  Regression (`0.518381515`) and independent Random Forest (`0.518205956`) were the descriptive
  leaders, but their paired improvements versus Logistic were only `0.008390596`
  (`[-0.024561612, 0.060292662]`) and `0.008215038`
  (`[-0.011754340, 0.026133447]`).
- Combined within-subject balanced accuracy ranged from `0.487378289` to `0.503456996`.
  Multi-output ElasticNet/Lasso led descriptively at `0.503456996`, with paired improvement
  `0.003443392` and interval `[-0.014362743, 0.021730064]`.
- Every candidate's pointwise paired 95% balanced-accuracy improvement interval versus Logistic
  Regression included zero. Exact-match accuracy was zero for all ten models in both protocols.
- Classifier pooled ECE was `0.248079` Logistic, `0.052025` Linear SVM, and `0.032203` Ridge
  Classifier cross-subject; within-subject values were `0.279373`, `0.064972`, and `0.056217`.
  Better calibration did not produce better thresholded reconstruction accuracy.
- The global-majority baseline score-MSE was `0.249715` cross-subject and `0.250100`
  within-subject, lower than every learned model. This comparison remains semantic-aware:
  classifiers use probability Brier score and regressors use clipped-output MSE.
- Regression clipping was substantial for several linear variants. Combined within-subject
  independent Ridge clipped `34.95%` of raw sample-pixel outputs and multi-output ElasticNet
  clipped `29.58%`; both Random Forest variants clipped none.
- Inference limits: intervals are pointwise and not multiplicity-adjusted; pooled ECE treats
  sample-pixel pairs descriptively; PLS remains exploratory; protocols are never averaged.
- Verification: comparison tests and notebook integration passed; all four figures were visually
  inspected; Ruff, lockfile, and diff checks passed; the full suite reported 334 passed with two
  pre-existing Python 3.13 multiprocessing warnings.
