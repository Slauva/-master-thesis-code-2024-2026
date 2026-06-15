# Classical Models For Random-Imagery Reconstruction

Status: in_progress
Last updated: 2026-06-15
Next stage: Final Review

## Goal

Create a model-agnostic random-imagery experiment framework and compare nine classical model
variants using the established target, leakage, evaluation-protocol, and artifact contracts.

## Scope

- Preserve `Data_Pattern/patt`, `type="random"`, the full `[0.5, 15.5)` epoch, and 36 row-major
  binary pixel targets.
- Preserve the fixed cross-subject and identity-overlapping bidirectional cross-trial protocols.
- Keep existing Logistic Regression APIs, CLI behavior, and schema-v1/v2 artifact reading
  compatible.
- Add independent Linear SVM and Ridge Classifier variants with grouped Platt calibration.
- Add independent and multi-output Ridge Regression, ElasticNet, and Random Forest variants.
- Add exploratory multi-output PLS Regression.
- Persist new schema-v3 artifacts under model-specific immutable run roots.
- Execute all nine variants for cross-subject and both cross-trial directions.

## Acceptance Criteria

- Shared experiment contracts are model-agnostic without changing existing Logistic Regression
  predictions or artifact compatibility.
- Every model emits finite float64 scores in `[0, 1]` with binary predictions at threshold `0.5`.
- Classifier scores use train-only grouped Platt calibration; regressor scores use documented
  clipping with clipping diagnostics.
- Independent models retain one fitted pipeline per pixel; multi-output models retain one fitted
  pipeline for all 36 targets.
- Every learned transform, selector, calibrator, and estimator is fitted without outer-test access.
- CLI can run, safely evaluate, compare, and immutably reuse model/protocol runs.
- Twenty-seven new real direction runs and the final comparison notebooks are executed and
  validated.
- Ruff, lockfile validation, the full test suite, notebook integration checks, and
  `git diff --check` pass.

## Stages

### 1. Common Experiment Framework - Completed

- Objective: Generalize contracts, registry, configuration, and protocol orchestration while
  retaining complete Logistic Regression compatibility.
- Deliverables:
  - model-agnostic `experiments/random_imagery` package;
  - typed model registry and shared experiment configuration;
  - reusable protocol runner contracts;
  - compatibility wrappers or aliases for existing Logistic Regression public APIs;
  - focused compatibility tests.
- Constraints:
  - do not implement the new estimators in this stage;
  - schema-v1/v2 readers and existing Logistic Regression CLI remain unchanged;
  - reference schema-v1/v2 predictions must reproduce exactly.
- Verification:
  - existing Logistic Regression tests;
  - common-framework contract and registry tests;
  - safe loading and summary of existing schema-v1/v2 runs;
  - trusted replay of the reference run without numerical change;
  - Ruff and `git diff --check`.
- Completion criteria: shared code can describe all planned model variants and execute the
  established protocol through a backend contract while Logistic Regression behavior remains
  unchanged.
- Review gate: Stop and wait for explicit user approval.

### 2. Calibrated Classifiers - Completed

- Objective: Implement independent Linear SVM and Ridge Classifier models with grouped,
  train-only Platt calibration.
- Deliverables: classifier backends, per-pixel grouped search, calibration contracts,
  diagnostics, and tests.
- Constraints: calibration must use outer-train OOF decision scores only; outer-test rows remain
  inaccessible until all fitting and calibration are complete.
- Verification: deterministic synthetic runs, score bounds, threshold invariants, and explicit
  test-access ordering checks.
- Completion criteria: both classifiers produce validated `(sample, 36)` calibrated scores and
  binary predictions.
- Review gate: Stop and wait for explicit user approval.

### 3. Regression Models - Completed

- Objective: Implement independent and multi-output Ridge, ElasticNet, and Random Forest plus
  exploratory multi-output PLS.
- Deliverables: regression backends, multi-target selector, clipping diagnostics, search results,
  and tests.
- Constraints:
  - Lasso is represented by ElasticNet `l1_ratio=1`;
  - multi-output models use one selector, pipeline, and hyperparameter set for all targets;
  - regression ties use lower clipped validation MSE, then configured candidate order.
- Verification: output shapes, topology, deterministic fitting, convergence, score bounds, and
  delayed test access.
- Completion criteria: all seven regression variants satisfy the shared model-result contract.
- Review gate: Stop and wait for explicit user approval.

### 4. Artifacts, Workflow, And CLI - Completed

- Objective: Add schema-v3 immutable model artifacts and terminal workflows.
- Deliverables:
  - model-aware atomic writer, safe reader, and trusted replay;
  - `random-imagery-models run`, `evaluate`, and `compare`;
  - equivalent module entry point and complete CLI tests.
- Constraints:
  - store one multi-output pipeline or 36 independent pipelines according to topology;
  - include model ID and topology in run identity;
  - never load joblib during safe metadata evaluation.
- Verification: round trip, exact replay, corruption rejection, duplicate refusal, reuse, and
  mixed schema-v2/v3 comparison.
- Completion criteria: all planned models can be trained and evaluated from the terminal without
  source edits.
- Review gate: Stop and wait for explicit user approval.

### 5. Full Real-Corpus Training - Completed

- Objective: Execute every new model under both established protocols.
- Deliverables:
  - executed `notebooks/5.2-classical-models-training.ipynb`;
  - 27 validated immutable direction runs;
  - notebook integration checks and experiment-memory updates.
- Constraints: the notebook calls reusable workflow code and re-executes through immutable reuse.
- Verification: expected 141/39 and 81/81 splits, leakage audits, execution marker, no cell errors,
  and complete artifact inventories.
- Completion criteria: every model/protocol direction has a validated source-backed run.
- Review gate: Stop and wait for explicit user approval.

### 6. Final Comparison - Awaiting Review

- Objective: Compare all new variants with the existing Logistic Regression reference without
  conflating evaluation protocols.
- Deliverables:
  - executed `notebooks/5.3-classical-models-comparison.ipynb`;
  - separate cross-subject and within-subject tables;
  - uncertainty, score-MSE, IoU, Hamming, calibration, and clipping diagnostics;
  - paired subject-bootstrap differences on compatible splits.
- Constraints:
  - do not average cross-subject and within-subject estimates;
  - classifier scores are calibrated probabilities; regressor scores are clipped continuous
    outputs and must not be described as probabilities.
- Verification: visual inspection, integration tests, Ruff, lockfile check, full pytest, and diff
  check.
- Completion criteria: the comparison is reproducible, uncertainty-aware, and scientifically
  qualified.
- Review gate: Stop and wait for explicit user approval.

## Decisions And Assumptions

- Planned model IDs are:
  `linear-svm-independent`, `ridge-classifier-independent`,
  `ridge-regression-independent`, `ridge-regression-multioutput`,
  `elastic-net-independent`, `elastic-net-multioutput`,
  `random-forest-independent`, `random-forest-multioutput`, and
  `pls-regression-multioutput`.
- Each model performs its own train-only feature-family screening.
- Selection uses mean thresholded balanced accuracy; regression ties use lower clipped validation
  MSE and then configured order.
- Linear SVM and Ridge Classifier use subject-grouped OOF Platt calibration.
- Regression predictions are clipped to `[0, 1]`; lower and upper clipping fractions are retained.
- `score_mse` means Brier score for calibrated classifiers and clipped MSE for regressors.
- Independent variants tune each pixel separately; multi-output variants tune one shared pipeline.
- Multi-output feature ranking averages fold-local `f_classif` percentile ranks across 36 targets
  and resolves ties by original feature index.
- Random Forest estimators use `n_jobs=1`; outer search owns parallel scheduling.
- PLS is implemented only as a multi-output exploratory model.
- New artifacts use schema version 3 under
  `artifacts/experiments/random-imagery/<model-id>/<config-hash>/`.

## Progress Log

- 2026-06-15: User requested the same leakage-controlled experiment plan for Linear SVM, Ridge
  Classifier, Ridge/ElasticNet/Random Forest regression, and optional PLS.
- 2026-06-15: User selected a common framework, grouped Platt calibration, both independent and
  multi-output strategies, one ElasticNet grid including Lasso, per-model feature screening, and
  complete real runs for both protocols.
- 2026-06-15: User explicitly approved implementation. Plan saved and Stage 1 marked In Progress.
- 2026-06-15: Stage 1 added the `experiments/random_imagery` package with a typed ten-entry model
  registry, shared configuration contracts, bounded score/prediction schemas, a backend-driven
  protocol runner, and a Logistic Regression compatibility backend.
- 2026-06-15: Moved the canonical random-imagery target/protocol, non-EEG baseline, and metric
  implementations into the common package. Existing `experiments.logistic_regression` modules are
  compatibility wrappers, and schema-v1/v2 artifact and CLI behavior remains unchanged.
- 2026-06-15: Common-runner parity tests reproduced the legacy Logistic Regression selected
  feature families, probabilities, predictions, metrics, and bootstrap samples exactly for
  synthetic cross-subject and bidirectional cross-trial protocols.
- 2026-06-15: Real schema-v1 run `f515948b6bf5af55` trusted replay reproduced stored `(39, 36)`
  probabilities and predictions bit-for-bit. Safe CLI evaluation passed for schema-v1 and
  schema-v2 run `4fcdf3c4fa5ef75a`.
- 2026-06-15: Verification passed: 70 experiment tests, `uv run ruff check .`,
  `uv lock --check`, `uv run pytest` (`286 passed`, two pre-existing multiprocessing warnings),
  and `git diff --check`. Stage 1 is Awaiting Review; Stage 2 has not started.
- 2026-06-15: User explicitly requested continuation, approving Stage 1. Stage 1 marked Completed
  and Stage 2 marked In Progress.
- 2026-06-15: Stage 2 added typed Linear SVM and Ridge Classifier configurations, model-specific
  train-only feature-family screening, per-pixel grouped grid search, grouped OOF decision-score
  generation, and one-dimensional Logistic Regression Platt calibrators.
- 2026-06-15: Each final classifier pipeline is fitted on all direction-training rows only after
  hyperparameter selection. Every calibration score comes from a cloned selected pipeline fitted
  without that score's grouped validation fold; outer-test features remain unavailable until all
  pixel pipelines and calibrators are complete.
- 2026-06-15: Synthetic cross-subject tests validated deterministic bounded float64 scores, exact
  `0.5` thresholding, grouped OOF fold coverage, estimator topology, and delayed test access for
  both classifiers. Additional bidirectional cross-trial smoke runs completed for both models.
- 2026-06-15: Verification passed: 76 experiment tests, `uv run ruff check .`,
  `uv lock --check`, `uv run pytest` (`292 passed`, two pre-existing multiprocessing warnings),
  and `git diff --check`. Stage 2 is Awaiting Review; Stage 3 has not started.
- 2026-06-15: User explicitly requested continuation, approving Stage 2. Stage 2 marked Completed
  and Stage 3 marked In Progress.
- 2026-06-15: Stage 3 added strict configurations and backends for independent and multi-output
  Ridge, ElasticNet, and Random Forest regression plus exploratory multi-output PLS.
- 2026-06-15: Independent variants use one grouped search and fitted pipeline per pixel.
  Multi-output variants use one shuffled subject-grouped fold partition, one fold-local
  multi-target selector, one search, and one fitted pipeline across all targets.
- 2026-06-15: Regression screening and grid selection maximize thresholded balanced accuracy,
  then minimize clipped validation MSE, then preserve configured candidate order. Screening
  estimator hyperparameters are explicit and independent of search-grid ordering.
- 2026-06-15: Raw regression predictions are retained long enough to compute lower/upper clipping
  fractions, then clipped into finite float64 `[0, 1]` scores with exact configured-threshold
  labels. Random Forest estimators retain `n_jobs=1`; outer search owns parallelism.
- 2026-06-15: Synthetic tests covered all seven variants, deterministic fitting, both topologies,
  exact 36-target outputs, multi-target ranking, Lasso inclusion, tie-breaking, clipping
  diagnostics, configured thresholds, and delayed outer-test access. Representative independent
  and multi-output models completed bidirectional cross-trial smoke runs.
- 2026-06-15: A target-only real-corpus audit validated shuffled five-fold multi-output grouped CV
  on all `(180, 36)` random-imagery targets from 33 subjects. Every train and validation fold
  retained both classes for every pixel; validation sizes were `39, 39, 39, 27, 36`.
- 2026-06-15: Verification passed: 98 experiment tests, `uv run ruff check .`,
  `uv lock --check`, `uv run pytest` (`316 passed`, two pre-existing multiprocessing warnings),
  and `git diff --check`. Stage 3 is Awaiting Review; Stage 4 has not started.
- 2026-06-15: User explicitly approved Stage 3. Stage 3 marked Completed and Stage 4 marked In
  Progress.
- 2026-06-15: Stage 4 added schema-v3 model-aware immutable artifacts with atomic publication,
  complete SHA-256 inventories, safe metadata/array loading, and an explicit trusted joblib replay
  boundary.
- 2026-06-15: Artifact identity now includes schema version, model ID, topology-bearing resolved
  configuration, protocol, and direction. Independent variants persist one pipeline per target;
  multi-output variants persist one shared pipeline.
- 2026-06-15: Safe loading validates pipeline metadata and manifested paths without deserializing
  joblib. Trusted replay verifies the feature family, ordered feature/channel names, and exact
  canonical test sample keys before reproducing stored scores and predictions.
- 2026-06-15: Added one public train-or-reuse workflow and equivalent
  `random-imagery-models` / `python -m experiments.random_imagery` commands for `run`, `evaluate`,
  and mixed schema-v2/v3 `compare`.
- 2026-06-15: Synthetic round trips covered independent calibrated classification, independent
  regression, and multi-output regression; exact trusted replay, corruption and unsafe-path
  rejection, duplicate refusal, complete-set reuse, and combined within-subject evaluation passed.
- 2026-06-15: Verification passed: 113 experiment tests, `uv run ruff check .`,
  `uv lock --check`, `uv run pytest` (`329 passed`, two pre-existing multiprocessing warnings),
  both CLI entry points, and `git diff --check`. Stage 4 is Awaiting Review; Stage 5 has not
  started.
- 2026-06-15: User requested continuation, explicitly approving Stage 4. Stage 4 marked Completed
  and Stage 5 marked In Progress.
- 2026-06-15: Added and executed `notebooks/5.2-classical-models-training.ipynb`. It calls the
  shared workflow for all nine models and both protocols, then re-executes every call through
  immutable reuse.
- 2026-06-15: Published and safely validated exactly 27 current schema-v3 direction runs under
  `artifacts/experiments/random-imagery/` (about 127 MiB): nine cross-subject and eighteen
  bidirectional cross-trial runs.
- 2026-06-15: Every cross-subject run retained the expected 141/39 rows with disjoint subjects.
  Every cross-trial direction retained 81/81 rows from 27 overlapping identities while sample
  keys, seeds, image payloads, and trial contracts remained leakage-safe.
- 2026-06-15: Real ElasticNet fitting exposed strict convergence failures. Independent ElasticNet
  converged with `max_iter=1_000_000`, `tol=1e-4`; multi-output ElasticNet required
  `max_iter=1_000_000`, `tol=1e-3`. Convergence warnings remain fatal rather than suppressed.
- 2026-06-15: Notebook validation confirmed finite metrics and intervals, 36 pipelines for every
  independent run, one pipeline for every multi-output run, one saved descriptive figure, no cell
  errors, and marker `CLASSICAL_MODELS_TRAINING_VERIFIED`.
- 2026-06-15: Verification passed: notebook integration test, visual figure inspection,
  `uv run ruff check .`, `uv lock --check`, `uv run pytest` (`330 passed`, two pre-existing
  multiprocessing warnings), and `git diff --check`. Stage 5 is Awaiting Review; Stage 6 has not
  started.
- 2026-06-15: User requested continuation, explicitly approving Stage 5. Stage 5 marked Completed
  and Stage 6 marked In Progress.
- 2026-06-15: Added a reusable protocol-comparison module that safely loads schema-v2/v3 runs,
  requires exact ordered sample-key, target, and subject compatibility, and applies the same
  2,000 subject-cluster bootstrap draws to every paired model/reference contrast.
- 2026-06-15: Executed `notebooks/5.3-classical-models-comparison.ipynb`. Cross-subject and
  combined bidirectional cross-trial results remain separate; all four diagnostic figures were
  visually inspected and the notebook marker `CLASSICAL_MODELS_COMPARISON_VERIFIED` passed.
- 2026-06-15: No candidate produced a pointwise paired 95% balanced-accuracy improvement interval
  excluding zero versus Logistic Regression. Descriptive leaders were independent Ridge
  Regression (`0.518381515`) cross-subject and multi-output ElasticNet/Lasso (`0.503456996`)
  within-subject.
- 2026-06-15: Grouped Platt calibration reduced pooled ECE relative to Logistic Regression, but
  thresholded accuracy remained near chance. The global-majority baseline retained lower
  score-MSE than every learned model in both protocols; regression clipping remained substantial
  for several linear variants.
- 2026-06-15: Verification passed: comparison and notebook integration tests, visual inspection,
  `uv run ruff check .`, `uv lock --check`, `uv run pytest` (`334 passed`, two pre-existing
  multiprocessing warnings), and `git diff --check`. Stage 6 is Awaiting Review.
