# Pixel-wise Logistic Regression For Random Imagery

Status: in_progress
Last updated: 2026-06-15
Next stage: 5 - Metrics And Executed Notebook (Awaiting Review)

## Goal

Implement 36 independent binary Logistic Regression models that reconstruct one 6x6 random
visual-imagery stimulus from one full 15-second `Data_Pattern/patt` EEG feature row.

## Scope

- Use only `type="random"` blocks from `Data_Pattern/patt`.
- Split subjects 80/20 with deterministic grouped splitting and keep the test set untouched until
  feature-family selection and hyperparameter search are complete.
- Select one common feature family on grouped train-only cross-validation, then tune one pipeline
  per pixel.
- Persist complete, versioned experiment runs under
  `artifacts/experiments/logistic-regression/<config-hash>/`.
- Document the final reproducible experiment in an executed notebook.

## Acceptance Criteria

- Targets are binary row-major arrays with shape `(180, 36)` and stable pixel names.
- The fixed split contains 141 train rows and 39 test rows from 26 and 7 disjoint subjects.
- Subjects, sample keys, seeds, and image payloads do not overlap between train and test.
- Learned transforms are fitted only inside train folds; test data are evaluated once.
- Thirty-six fitted pipelines and finite predictions are persisted with validated provenance.
- The final notebook executes without errors and repository checks pass.

## Stages

### 1. Contracts, Targets, And Split - Completed

- Objective: Define the experiment configuration, target matrix, metadata, grouped split, leakage
  checks, and non-EEG baselines.
- Deliverables: typed config and schemas, target/split builders, majority, pixel-frequency, and
  seeded Bernoulli baselines, focused tests.
- Constraints: no feature extraction, feature selection, Logistic Regression fitting, artifacts,
  or notebook results in this stage.
- Verification: focused experiment tests, deterministic real-corpus split audit, Ruff.
- Completion criteria: target and metadata contracts are strict; split 42 reproduces 141/39 rows
  with all 36 tasks containing both classes and no configured leakage overlap.
- Review gate: Stop and wait for explicit user approval.

Implemented:

- Added a frozen, extra-forbidding OmegaConf/Pydantic experiment configuration covering dataset
  selection, grouped split, future train-only CV/screening/grid parameters, prediction threshold,
  bootstrap settings, and artifact root/schema.
- Added a versioned deterministic experiment-config hash that includes the artifact schema and
  feature-extractor version.
- Added immutable target, split, leakage-audit, and baseline-prediction schemas with strict dtype,
  shape, binary-value, partition, and probability validation.
- Added deterministic row-major 6x6 target construction from `RandomSample.img`, stable pixel names
  `pixel_r0_c0` through `pixel_r5_c5`, canonical sample metadata, and SHA-256 image fingerprints.
- Added subject-wise `GroupShuffleSplit` plus explicit overlap checks for subjects, sample keys,
  random seeds, and full image payloads. Optional class validation requires both labels in every
  train and test pixel task.
- Added three non-EEG baselines: global majority, per-pixel training frequency, and seeded
  Bernoulli sampling from per-pixel training frequencies.

Verification:

- `uv run pytest tests/experiments -q`: 24 passed.
- Real `Data_Pattern/patt`, `type="random"` audit: targets `(180, 36)`, train/test rows `141/39`,
  train/test subjects `26/7`, held-out subjects `(9, 10, 16, 18, 20, 28, 33)`.
- Train positive counts range from 59 to 81 and test positive counts from 14 to 26; all 36 tasks
  contain both classes in both partitions.
- No overlap was found in subjects, canonical sample keys, seeds, or image fingerprints.
- All three baseline outputs have shape `(39, 36)` and deterministic seeded behavior.
- `uv run ruff check .`: passed.
- `uv run pytest`: 236 passed, 2 skipped; two existing Python 3.13 multiprocessing `fork()`
  deprecation warnings remain.
- `git diff --check`: passed.

### 2. Common Feature-Family Selection - Completed

- Objective: Select one common feature family using grouped train-only cross-validation.
- Deliverables: feature alignment, candidate-family screening, deterministic tie-breaking, result
  schemas and tests.
- Constraints: test rows remain untouched; all variance filtering, univariate selection, and
  scaling occur inside folds.
- Verification: synthetic leakage tests and real train-only screening checks.
- Completion criteria: every candidate is evaluated identically and one common feature set is
  selected from mean per-pixel balanced accuracy.
- Review gate: Stop and wait for explicit user approval.

Implemented:

- Added strict schemas for aligned train-only feature families, per-pixel grouped folds, candidate
  scores, selected-feature counts, and deterministic final selection.
- Added canonical-key feature alignment that loads only outer-train rows and rejects windowed or
  mismatched feature sets. All nine candidates are flattened from each loaded feature set once.
- Added deterministic per-pixel `StratifiedGroupKFold` plans. Every candidate uses the same folds
  for a given pixel, and every fold checks class support plus subject disjointness.
- Added a fold-local pipeline with variance filtering, capped univariate ANOVA selection,
  standardization, and fixed balanced L2 Logistic Regression.
- Added stable handling for constant and perfectly separating ANOVA features plus deterministic
  candidate-order tie-breaking.
- Added synthetic tests proving outer-test keys are not loaded and learned transforms are fitted
  only on fold-training rows.

Real train-only screening:

- Shapes ranged from `(141, 819)` for `time` to `(141, 16128)` for each local-pattern family.
- Mean per-pixel five-fold balanced accuracies were: `time` 0.508035, `spectral` 0.501098,
  `time+spectral` 0.505047, `covariance` 0.494298, `correlation` 0.490803,
  `log_covariance` 0.500776, `lndp` 0.501745, `lgp` 0.510085, and `lbp` 0.515542.
- `lbp` was selected by the predefined maximum-mean rule. The margin is small and all screening
  scores are near chance, so this is a procedural train-only choice rather than evidence of useful
  held-out performance.
- A second complete cached screening run reproduced every candidate score within `5e-10`.
- Feature manifests were absent for all 39 outer-test keys after both runs.

Verification:

- `uv run pytest tests/experiments -q`: 29 passed.
- Real fold audit: all 180 pixel-fold combinations had both classes in train and validation with
  disjoint subjects; validation folds contained 24-33 rows from 4-7 subjects.
- `uv run ruff check .`: passed.
- `uv run pytest`: 243 passed; two existing Python 3.13 multiprocessing `fork()` deprecation
  warnings remain.
- `git diff --check`: passed.

### 3. Per-pixel Grid Search - Completed

- Objective: Tune and fit 36 independent Logistic Regression pipelines.
- Deliverables: grouped GridSearchCV orchestration, predictions, selected-feature/coefficient
  extraction, metrics and tests.
- Constraints: fixed threshold 0.5; test set evaluated only after all train-only decisions.
- Verification: fold-class checks, convergence checks, deterministic predictions and finite output.
- Completion criteria: exactly 36 fitted pipelines produce `(39, 36)` probabilities and labels.
- Review gate: Stop and wait for explicit user approval.

Implemented:

- Added strict schemas for aligned feature partitions, grid candidates, selected
  hyperparameters, fitted pixel pipelines, selected original-feature indices, coefficients, and
  final test predictions.
- Added a staged orchestration that loads only the 141 outer-train `lbp` rows, completes all 36
  train-only grouped searches, and only then loads the 39 outer-test rows.
- Added 36 independent `GridSearchCV` runs over 64 combinations of `k`, `C`, L1/L2, and class
  weight. Every pixel reuses its Stage 2 subject-wise folds and balanced-accuracy scoring.
- Kept variance filtering, ANOVA selection, scaling, and Logistic Regression inside each
  `Pipeline`. Temporary joblib caching avoids repeated fold transforms and is removed from each
  returned fitted pipeline.
- Mapped L1/L2 to sklearn 1.9's non-deprecated `l1_ratio=1/0` contract and converted selected
  feature support back to original `lbp` column indices and names.
- Added synthetic tests for test-access ordering, fold-local learned transforms, complete fitted
  model metadata, deterministic probabilities, predictions, CV scores, and hyperparameters.

Real run:

- Completed all 36 searches and one outer-test prediction pass in 43.958 s with `n_jobs=-1`.
- Produced finite probability and label matrices with shape `(39, 36)` and finite coefficients
  for every fitted pipeline.
- Mean best train-only CV balanced accuracy was 0.579192, ranging from 0.500000 to 0.684022.
- Mean outer-test per-pixel balanced accuracy was 0.509991 with standard deviation 0.085920 and
  range 0.360963-0.658730.
- Selected `k`: 25 for 6 pixels, 50 for 9, 100 for 8, and 250 for 13.
- Selected `C`: 0.01 for 6 pixels, 0.1 for 11, 1.0 for 12, and 10.0 for 7.
- Selected penalty: L1 for 17 pixels and L2 for 19. Class weight was `None` for 22 and `balanced`
  for 14.
- Final `liblinear` iteration counts ranged from 0 to 13. Zero iterations are valid when the
  initial solution satisfies the optimizer; the result schema was corrected accordingly.
- The gap between mean best CV and mean test score, plus a near-chance mean test score, is evidence
  of substantial selection optimism and weak generalization, not successful image reconstruction.

Verification:

- `uv run pytest tests/experiments -q`: 32 passed.
- Real one-pixel train-only rerun reproduced pixel 0's `k=50`, `C=1`, L2, no-class-weight choice
  and CV balanced accuracy 0.539780 without touching outer-test rows.
- `uv run ruff check .`: passed.
- `uv run pytest`: 246 passed; two existing Python 3.13 multiprocessing `fork()` deprecation
  warnings remain.
- `git diff --check`: passed.

Known boundary:

- Stage 3 returns the complete fitted result in memory but does not write the official experiment
  run. Atomic persistence, hashes, immutable manifests, and joblib validation remain Stage 4.

### 4. Experiment Artifacts - Completed

- Objective: Persist and validate complete reproducible experiment runs.
- Deliverables: atomic writer/reader, manifest and hashes, config/environment/split/results files,
  36 joblib pipelines, corruption and round-trip tests.
- Constraints: manifest written last; valid runs are immutable without explicit overwrite; only
  trusted local joblib files may be loaded.
- Verification: round-trip, missing-file, hash-mismatch, duplicate-run and prediction-reproduction
  tests.
- Completion criteria: a saved run can reproduce recorded predictions and rejects incomplete or
  changed content.
- Review gate: Stop and wait for explicit user approval.

Implemented:

- Added a directory-level atomic experiment writer. It builds the entire run under a hidden
  sibling temporary directory, writes `manifest.json` last, fsyncs payloads, and publishes with an
  atomic rename to the experiment config hash.
- Added immutable-by-default behavior: an existing config-hash run raises `FileExistsError` unless
  the resolved config explicitly enables overwrite.
- Persisted resolved config, Python/platform/package versions, git commit/dirty state, outer split,
  all per-pixel grouped folds, full feature names, feature config hash, screening scores, complete
  64-candidate grid summaries, selected supports/coefs, train/test targets, predictions,
  probabilities, per-pixel test balanced accuracy, and one joblib pipeline per pixel.
- Added exact file inventory, size, and SHA-256 verification for every payload file. The manifest
  itself is written last and excluded from its own hash inventory.
- Added an explicit trust boundary: metadata and hashes are validated first, but joblib pipelines
  load only when the caller passes `trusted=True`. Pipeline paths must be relative, live directly
  under `pipelines/`, use `.joblib`, and appear in the manifest.
- Added replay support that validates feature family, full feature-name order, and canonical
  test sample keys before reproducing probabilities and labels.
- Added focused tests for atomic round trip, exact prediction replay, immutable duplicate refusal,
  missing files, hash/size corruption, unsafe pipeline paths, and untrusted joblib refusal.
- Added `artifacts/experiments/` to `.gitignore`.

Real artifact:

- Reproduced screening (`lbp`) and all 36 per-pixel models, then atomically published
  `artifacts/experiments/logistic-regression/f515948b6bf5af55/`.
- The run occupies about 14 MiB and contains 49 files: 48 manifest-tracked payloads plus
  `manifest.json`.
- Loaded all 36 hash-validated pipelines and reproduced the stored `(39, 36)` probabilities
  bit-for-bit and predictions exactly from the aligned test features.
- The persisted mean outer-test balanced accuracy is 0.509990919, matching Stage 3.
- Recorded environment: Python 3.13.11, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.9.0,
  joblib 1.5.3, MNE 1.11.0, Pydantic 2.12.5, and OmegaConf 2.3.0.
- Provenance records git commit `1ca50bf23fdbffb79609a80bacb2f7884e4ac8bc` with
  `git_dirty=true`; payload integrity is complete, but final thesis provenance should point to a
  committed revision.

Verification:

- `uv run pytest tests/experiments -q`: 36 passed.
- Real run manifest validation and pipeline replay: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: 251 passed; two existing Python 3.13 multiprocessing `fork()` deprecation
  warnings remain.
- `git diff --check`: passed.

### 5. Metrics And Executed Notebook - Awaiting Review

- Objective: Run, visualize, and document the complete baseline experiment.
- Deliverables: aggregate/per-pixel metrics, subject bootstrap confidence intervals, figures and
  executed `notebooks/5.0-logistic-regression-random-pixels.ipynb`.
- Constraints: notebook calls reusable tested modules and reads the persisted run; it does not own
  training logic.
- Verification: visual inspection, notebook integration test, Ruff, full pytest, diff check.
- Completion criteria: notebook reports baselines, feature screening, grid search, reconstruction
  examples, feature interpretation, limitations and artifact provenance.
- Review gate: Stop and wait for explicit user approval.

Implemented:

- Added tested aggregate and per-pixel evaluation for balanced accuracy, macro F1, Brier score,
  bit accuracy, exact 6x6 match accuracy, and Hamming distance.
- Added deterministic percentile cluster bootstrap for mean per-pixel balanced accuracy. Subjects
  are sampled as complete clusters with replacement; the rare draw that removes a target class
  from any pixel is rejected rather than changing the balanced-accuracy definition.
- Added and executed `notebooks/5.0-logistic-regression-random-pixels.ipynb`. It validates and
  reads the immutable run, evaluates three non-EEG baselines, visualizes subject-bootstrap
  uncertainty, screening scores, CV/test optimism, the 6x6 per-pixel score map, deterministic
  reconstruction examples, and descriptive LBP selection frequencies.
- Added a notebook integration test requiring complete execution, no error outputs, five stored
  figures, the persisted run ID, reusable evaluation APIs, and the final validation marker.

Final results:

- Logistic Regression mean per-pixel balanced accuracy is `0.509990919`; the 95% subject-cluster
  bootstrap interval is `[0.496383660, 0.521077288]` from 2,000 valid resamples and includes
  chance `0.5`.
- Mean macro F1 is `0.499652645`, bit accuracy is `0.514245014`, mean Brier score is
  `0.333966692`, exact 6x6 match accuracy is `0`, and mean Hamming distance is `17.487179` pixels.
- Pixel-frequency baseline has balanced accuracy `0.5`, higher bit accuracy `0.524216524`, lower
  Brier score `0.250529325`, and lower mean Hamming distance `17.128205`; no baseline produced an
  exact reconstruction.
- The model's mean best inner-CV balanced accuracy `0.579192` exceeds held-out performance by
  about `0.0692`. Per-pixel scores and reconstruction examples show no coherent spatial recovery.
- LBP channel/code summaries are descriptive only because features were standardized and each
  pixel used an independently regularized model; they are not physiological importance estimates.

Verification:

- Executed notebook: 9 code cells, 5 stored PNG figures, no error outputs, and
  `LOGISTIC_REGRESSION_RANDOM_PIXELS_VERIFIED`.
- Visual inspection of all five figures: passed.
- `uv run pytest tests/experiments tests/test_logistic_regression_notebook.py -q`: 43 passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: 257 passed; two existing Python 3.13 multiprocessing `fork()` deprecation
  warnings remain.
- `git diff --check`: passed.

## Decisions And Assumptions

- `Data_Pattern/patt` is the sole recording family; `Data_Train/exec` and geometric blocks are
  excluded.
- One full `[0.5, 15.5)` imagery epoch is one prediction row; temporal windows are excluded.
- Split groups are subject IDs. `GroupShuffleSplit(test_size=0.2)` holds out 20% of groups, not
  exactly 20% of rows.
- Random state is 42 for the outer split, inner folds, stochastic baseline, and bootstrap.
- The primary model-selection metric is balanced accuracy and the final binary threshold is 0.5.
- One common feature family is selected for all pixels; per-pixel `k`, `C`, penalty, and class
  weight are tuned later.
- Previous feature-extraction Stage 5 remains a separate awaiting-review gate.

## Progress Log

- 2026-06-14: User explicitly approved the implementation plan including persisted artifacts.
- 2026-06-14: Stage 1 started.
- 2026-06-14: Stage 1 implemented and verified; awaiting explicit user approval.
- 2026-06-15: User asked to continue the plan; Stage 1 approved and marked completed.
- 2026-06-15: Stage 2 started.
- 2026-06-15: Stage 2 implemented and verified; `lbp` selected by train-only grouped CV and the
  stage is awaiting explicit user approval.
- 2026-06-15: User asked to continue; Stage 2 approved and marked completed.
- 2026-06-15: Stage 3 started.
- 2026-06-15: Stage 3 implemented and verified; 36 `lbp` pipelines produced finite outer-test
  predictions and the stage is awaiting explicit user approval.
- 2026-06-15: User asked to continue; Stage 3 approved and marked completed.
- 2026-06-15: Stage 4 started.
- 2026-06-15: Stage 4 implemented and verified; immutable run `f515948b6bf5af55` reproduces all
  recorded predictions and the stage is awaiting explicit user approval.
- 2026-06-15: User asked to continue; Stage 4 approved and marked completed.
- 2026-06-15: Stage 5 started.
- 2026-06-15: Stage 5 implemented and verified; the executed notebook reports a near-chance
  held-out result with a subject-bootstrap interval containing 0.5, and the stage is awaiting
  explicit final approval.
