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
- Use only `Data_Pattern/patt` records with `type="random"` for the primary pixel-wise Logistic
  Regression reconstruction baseline. One full `[0.5, 15.5)` imagery epoch maps to one 36-pixel
  target row.
- Hold out subjects with `GroupShuffleSplit(test_size=0.2, random_state=42)`. Interpret the 80/20
  ratio as a proportion of subject groups, not rows; the current corpus yields 141/39 rows from
  26/7 subjects.
- Reject outer splits that overlap in subject, canonical sample key, random seed, or complete image
  payload, and require both binary classes in every train and test pixel task.
- Use balanced accuracy as the primary train-only model-selection metric and 0.5 as the fixed final
  decision threshold. Select one common feature family before per-pixel hyperparameter tuning.

## 2026-06-15

- Screen common feature families separately for each pixel with deterministic five-fold
  `StratifiedGroupKFold` on the 141 outer-train rows, using subject ID as the group. Reuse the same
  pixel-specific folds for every candidate family.
- Fit variance filtering, capped ANOVA `SelectKBest`, standardization, and the fixed balanced L2
  Logistic Regression screening model only on each fold's training rows.
- Rank feature families by the unweighted mean of per-pixel mean fold balanced accuracy and break
  exact ties by configured candidate order.
- The current train-only screening selects `lbp` for Stage 3. Keep the outer-test feature rows and
  labels untouched until all per-pixel hyperparameters are selected and final models are fitted.
- Tune each pixel independently with the same pixel-specific grouped folds used for screening.
  Search `k in {25, 50, 100, 250}`, `C in {0.01, 0.1, 1, 10}`, L1/L2, and
  `class_weight in {None, balanced}` inside a variance-filter, ANOVA-selector, scaler, and
  `liblinear` pipeline.
- Represent sklearn 1.9 L1/L2 Logistic Regression through `l1_ratio=1/0` instead of the deprecated
  `penalty` constructor parameter. Preserve the scientific parameter names as `l1` and `l2` in
  experiment schemas and reports.
- Complete all 36 train-only grid searches before loading outer-test feature rows or computing any
  outer-test predictions. Parallel `n_jobs=-1` changes only execution scheduling, not folds,
  candidate order, seeds, or tie-breaking.
- Store experiment runs under the versioned config hash and publish the complete run directory
  atomically only after every payload file is durable and `manifest.json` has been written last.
- Treat valid experiment runs as immutable by default. Refuse duplicate config-hash writes unless
  overwrite is explicitly enabled in the resolved experiment config.
- Record the exact relative file inventory, byte size, and SHA-256 for every config, metadata,
  array, and pipeline payload. Reject missing, unexpected, size-changed, or hash-changed files.
- Never load persisted joblib pipelines implicitly. Require explicit `trusted=True` after manifest
  validation, and restrict pipeline paths to manifested `.joblib` files directly under
  `pipelines/`.
- Reproduce persisted predictions only after validating the selected feature family, complete
  feature-name/channel order, and canonical outer-test sample keys.
- Evaluate final 6x6 reconstruction with mean per-pixel balanced accuracy as the primary metric,
  plus macro F1, Brier score, bit accuracy, exact-match accuracy, and mean Hamming distance.
- Quantify held-out uncertainty with a percentile cluster bootstrap that resamples complete test
  subjects with replacement. Reject the rare resample that removes either class from any pixel so
  balanced accuracy retains the same 36-task definition.
- Treat selected standardized LBP coefficients and selection frequencies as descriptive model
  diagnostics only, not as physiological channel importance or causal evidence.
- Treat binary value `1` as foreground for reconstruction IoU. Compute sample IoU independently
  for each 6x6 image, assign `1.0` to an empty-target/empty-prediction sample, and compute micro
  IoU from foreground intersections and unions pooled across all evaluated samples and pixels.
- Report normalized Hamming loss alongside bit accuracy and mean Hamming distance, enforcing
  `hamming_loss = 1 - bit_accuracy = mean_hamming_distance / 36`.
- Keep IoU and Hamming loss strictly in held-out evaluation and reporting; balanced accuracy
  remains the primary feature-screening and hyperparameter-selection metric.
- Represent cross-subject and within-subject evaluation with separate typed direction contracts;
  do not relax the subject-disjoint `SubjectSplit` schema to accommodate identity overlap.
- Define the secondary protocol as `identity-overlapping bidirectional cross-trial`. Eligible
  identities must contain both Trial 1 and Trial 2; run Trial 1 -> Trial 2 and Trial 2 -> Trial 1
  independently and retain excluded identities as explicit provenance.
- Permit subject identity overlap only for the cross-trial protocol. Sample keys, trial numbers,
  random seeds, and complete image payloads must remain disjoint between a direction's train and
  test rows.
- Repeat feature-family screening, grouped inner CV, and all per-pixel hyperparameter searches
  independently inside each direction's training rows. Load that direction's test features only
  after all fitted models are complete.
- Combine within-subject directions only after both independent predictions exist. For combined
  uncertainty, resample whole identities so all six out-of-trial random-imagery rows for each
  selected real-corpus subject remain in one bootstrap cluster.
- Use artifact schema version 2 for new protocol-aware runs and include protocol plus direction in
  the immutable run hash. Preserve schema-v1 hash semantics for the existing reference run.
- Persist each within-subject direction independently, including baseline probabilities and
  predictions, so safe evaluation can recompute the exact combined metrics without rewriting
  either run.
- Keep metadata/array evaluation separate from trusted pipeline replay. Schema-v2 reuse must
  validate the full inventory and recompute model metrics, baseline metrics, and subject bootstrap
  summaries without loading joblib.
- Refuse duplicate CLI training runs and destructive overwrite. `--reuse-existing` is valid only
  when every expected protocol direction exists, validates, and matches the configured feature
  hash.
- Use standard-library `argparse` and equivalent installed `logistic-regression` and
  `python -m experiments.logistic_regression` entry points. Parse repeatable dotted overrides
  through OmegaConf rather than ad hoc scalar coercion.
- Keep train-or-reuse orchestration in one public `execute_evaluation_protocol` workflow shared by
  CLI and notebooks. Notebooks may format persisted results but must not duplicate splitting,
  fitting, persistence, reuse validation, or metric computation.
- Compare cross-subject, each cross-trial direction, and combined cross-trial uncertainty in one
  figure, but report cross-subject and within-subject tables separately. Never average protocols
  or describe their difference as a pure model effect because their populations and
  generalization targets differ.
- Keep model-agnostic random-imagery targets, evaluation protocols, baselines, metrics,
  configuration primitives, and orchestration under `experiments/random_imagery`. Preserve
  `experiments/logistic_regression` module paths as compatibility wrappers where implementations
  have moved.
- Describe every random-imagery estimator through a typed registry entry containing model ID,
  estimator family, independent or multi-output topology, classifier or regressor task, and score
  semantics.
- Require every model backend to complete training and feature-family selection before the common
  runner materializes outer-test features. Backends return bounded float64 scores and exact
  threshold-derived int8 predictions through one shared contract.
- Use `score_mse` as the model-agnostic name for squared error of bounded scores. Existing Logistic
  Regression `mean_brier_score` and `per_pixel_brier_score` fields remain intact and expose
  `mean_score_mse` and `per_pixel_score_mse` aliases for compatibility.
- For Linear SVM and Ridge Classifier, perform model-specific feature-family screening with the
  estimator's native zero decision threshold and balanced accuracy on the established
  pixel-specific grouped folds.
- Select each classifier pipeline by grouped balanced-accuracy grid search, then clone the selected
  pipeline into every grouped fold to generate one OOF decision score per outer-train row. Fit the
  scalar Logistic Regression Platt calibrator only on those OOF scores and labels.
- Fit one final selected base pipeline per pixel on all direction-training rows. Apply its decision
  scores to the fitted Platt calibrator at prediction time; never expose outer-test rows to feature
  selection, hyperparameter search, OOF score generation, or calibration.
- Retain OOF decision scores, OOF fold assignments, sigmoid coefficient/intercept, selected
  supports, base coefficients, and candidate CV summaries in the fitted classifier payload for
  later schema-v3 artifact persistence.
- Use the established pixel-specific `StratifiedGroupKFold` contracts for independent regressors.
  Use one shuffled `GroupKFold` partition by subject for multi-output regressors, and reject any
  fold where a target lacks either class in train or validation.
- Keep regression feature-family screening estimator parameters explicit in configuration and
  separate from search-grid order. Grid order affects only the final deterministic tie-break.
- Select regression feature families and hyperparameters by maximum mean thresholded balanced
  accuracy. Resolve exact ties by lower clipped validation MSE and then configured candidate order.
- For multi-output feature selection, compute fold-local `f_classif` scores for every target,
  convert each target's scores into deterministic percentile ranks, average ranks across targets,
  and resolve remaining ties by original feature index.
- Use sklearn's native multi-output `Ridge`, `ElasticNet`, and `RandomForestRegressor` behavior with
  one shared pipeline and hyperparameter set. Do not substitute `MultiTaskElasticNet`, whose
  mixed-norm sparsity is a different model.
- Represent Lasso inside the ElasticNet grid with `l1_ratio=1.0` and require every ElasticNet
  configuration to include that candidate.
- Apply external standardization before Ridge, ElasticNet, and PLS; configure PLS with
  `scale=False` to avoid a second scaling pass. Random Forest uses no scaler.
- Keep every Random Forest estimator at `n_jobs=1`; only the outer grouped grid search may
  parallelize candidate and fold evaluation.
- Measure fractions of raw regression outputs below zero and above one before clipping. Expose
  only clipped float64 `[0, 1]` scores and exact configured-threshold int8 labels to the common
  runner; clipped regression scores must not be described as probabilities.
- Use artifact schema version 3 for non-reference model runs under
  `artifacts/experiments/random-imagery/<model-id>/<config-hash>/`. Include model ID, complete
  resolved model configuration, protocol, and direction in the immutable hash.
- Persist 36 independently fitted pipelines for independent topology and exactly one fitted
  pipeline for multi-output topology. Persist grouped Platt calibration coefficients as validated
  JSON metadata rather than embedding calibrators in the base classifier pipelines.
- Publish schema-v3 runs atomically only after every payload is flushed and inventoried. Refuse
  duplicate publication and destructive overwrite.
- Keep schema-v3 safe loading free of joblib deserialization while still validating pipeline
  counts, relative paths, manifest membership, file hashes, arrays, metrics, and bootstrap
  summaries. Permit pipeline loading only through an explicit trusted mode.
- Require trusted replay to match selected feature blocks, complete ordered feature/channel names,
  and canonical test sample keys before calling persisted pipelines.
- Share one `execute_model_protocol` train-or-reuse workflow between terminal and future notebook
  callers. Reuse is valid only when the complete protocol direction set exists and each run
  matches both the resolved model configuration and feature configuration hash.
- Compare schema-v2 and schema-v3 direction runs only when protocol, direction, and ordered test
  sample keys match exactly; do not silently align or aggregate incompatible evaluation rows.
- Use `max_iter=1_000_000` with `tol=1e-4` for real independent ElasticNet runs. The approved grid
  is retained unchanged, including low regularization and `l1_ratio=1.0`; convergence warnings
  remain fatal.
- Use `max_iter=1_000_000` with `tol=1e-3` for real multi-output ElasticNet runs. The native
  multi-target `ElasticNet` cyclic solver did not satisfy `tol=1e-4` even at one million
  iterations on one grouped fold; the model-specific tolerance is explicit in configuration
  rather than suppressing the warning or silently dropping a candidate.
- For final model comparison, require exact ordered test sample keys, targets, and subject IDs
  across every candidate and the Logistic Regression reference. Reject rather than silently align
  incompatible runs.
- Keep cross-subject and combined bidirectional cross-trial comparisons separate. Combine the two
  cross-trial directions only after confirming disjoint sample keys, then keep all six rows from a
  sampled identity together in the subject-cluster bootstrap.
- Use the same 2,000 accepted subject-bootstrap draws for every model within a protocol. Report
  paired improvement with positive always meaning better: candidate minus Logistic for balanced
  accuracy and IoU, Logistic minus candidate for score-MSE and Hamming loss.
- Treat paired 95% intervals as pointwise exploratory intervals without multiplicity adjustment.
  Do not convert descriptive ranks or isolated absolute intervals into model-superiority claims.
- Assess calibration only for classifier probabilities using pooled sample-pixel fixed-bin ECE.
  For regressors, report pre-clipping lower/upper fractions and clipped score-MSE; never interpret
  clipped continuous outputs as calibrated probabilities.
