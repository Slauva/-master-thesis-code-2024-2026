# PyTorch Spectral Models For Random-Imagery Reconstruction

Status: in_progress
Last updated: 2026-06-16
Next stage: 6 - Final Comparison (awaiting review)

## Goal

Port the supplied ARL EEGModels TensorFlow architectures to PyTorch and run a reproducible,
leakage-controlled random-imagery reconstruction study over the established FFT, Morlet,
Superlet, and STFT representations.

## Scope

- Preserve `Data_Pattern/patt`, `type="random"`, the `[0.5, 15.5)` imagery epoch, 36 row-major
  binary pixel targets, and the established evaluation protocols.
- Port EEGNet, DeepConvNet, ShallowConvNet, EEGNet-SSVEP, and EEGNet-v1 to PyTorch.
- Use EEGNet, DeepConvNet, and ShallowConvNet in the full experiment.
- Treat every architecture/preprocessing pair as a separate model: 12 primary Torch variants.
- Use one multi-label model with 36 logits rather than 36 independent networks.
- Train a three-seed ensemble for every model and evaluation direction.
- Compare Torch models with Logistic Regression, all nine classical variants, and canonical
  non-EEG baselines.
- Remove `eegnet-tesnorflow.py` only after the PyTorch ports and upstream attribution are verified.

## Acceptance Criteria

- Spectral inputs are computed from the imagery crop before transformation and retain exact
  canonical sample-key alignment.
- Outer-test data is inaccessible during normalization, epoch selection, and all model fitting.
- Every model returns finite float64 sigmoid scores in `[0, 1]` and exact threshold-0.5 int8
  predictions with shape `(sample, 36)`.
- Three final seeds are trained on the complete outer-training partition and their probabilities
  are averaged for evaluation.
- Cross-subject and combined bidirectional cross-trial results remain separate.
- Thirty-six real Torch direction artifacts are produced and safely validated.
- Final comparison includes uncertainty, multiplicity-aware contrasts, calibration, runtime,
  parameter count, and seed sensitivity.
- Ruff, lockfile validation, full pytest, notebook integration checks, visual inspection, and
  `git diff --check` pass.

## Stages

### 1. Spectral Input Contract - Completed

- Objective: Build a crop-aware, leakage-safe Torch spectral input layer for all four established
  preprocessing methods.
- Deliverables:
  - crop-before-transform spectral dataset and immutable cache;
  - strict sample-key, channel, frequency, time-axis, shape, dtype, and scaling contracts;
  - FFT input `[batch, 1, electrode, frequency]`;
  - Morlet/Superlet/STFT input `[batch, frequency, electrode, time]`;
  - numerically stable log-power transform and train-only frequency-bin z-score state.
- Constraints:
  - crop exactly `[0.5, 15.5)` before preprocessing;
  - use EEG only; do not add EOG inputs;
  - validation and test normalization may use only statistics fitted on direction-training rows;
  - do not reuse full-recording spectral caches as imagery-crop inputs.
- Verification:
  - expected real shapes: FFT `(63, 39)`, Morlet `(63, 39, 49)`, Superlet `(63, 39, 46)`,
    and STFT `(63, 39, 51)`;
  - cache hit, invalidation, corruption, and source-signature tests;
  - exact target/sample alignment and explicit delayed-test-access checks;
  - Ruff and focused pytest.
- Completion criteria: every preprocessing method yields deterministic model-ready tensors without
  fitting or reading any outer-test-derived state.
- Review gate: Stop and wait for explicit user approval.

### 2. PyTorch Architecture Ports - Completed

- Objective: Port all five supplied architectures while defining scientifically explicit spectral
  adaptations.
- Deliverables:
  - PyTorch EEGNet, DeepConvNet, ShallowConvNet, EEGNet-SSVEP, and EEGNet-v1 modules;
  - one 36-logit multi-label head per model;
  - max-norm projection utilities;
  - retained upstream attribution, citations, and complete CC0/Apache-2.0 license notice;
  - removal of `eegnet-tesnorflow.py` after verification.
- Constraints:
  - preserve source filter counts, depthwise/separable blocks, nonlinearities, dropout, and
    max-norm intent;
  - use same padding and adaptive global pooling where short spectral axes make the original
    fixed valid-convolution geometry impossible;
  - describe the implementations as spectral adaptations, not numerically identical TensorFlow
    translations;
  - EEGNet-SSVEP and EEGNet-v1 are ported and tested but excluded from full training.
- Verification:
  - parameterized forward/backward tests over all four input geometries;
  - finite logits and gradients, exact output shape, depthwise grouping, max-norm enforcement,
    deterministic initialization, and parameter-count snapshots;
  - CPU and CUDA smoke tests.
- Completion criteria: all five models are reusable typed PyTorch modules, and the three primary
  architectures train on every spectral representation.
- Review gate: Stop and wait for explicit user approval.

### 3. Leakage-Safe Training - Completed

- Objective: Add deterministic grouped model selection and three-seed ensemble training.
- Deliverables:
  - strict Torch experiment configuration;
  - trainer, validation history, checkpoint state, epoch-selection diagnostics, and ensemble
    prediction contract;
  - `BCEWithLogitsLoss` with train-only per-pixel positive weights.
- Constraints:
  - AdamW with `lr=1e-3`, `weight_decay=1e-4`;
  - batch size 16, float32, no AMP, gradient clipping at `1.0`;
  - maximum 300 epochs;
  - deterministic three-fold subject-grouped validation with one selection seed;
  - choose best epoch by mean per-pixel balanced accuracy, then lower BCE;
  - early stopping `patience=30`, `min_delta=1e-4`;
  - final epoch count is the median of the three fold-best epochs;
  - final seeds are `42`, `43`, and `44`, each fitted on all outer-training rows;
  - ensemble score is the mean sigmoid probability; threshold remains `0.5`.
- Verification:
  - deterministic synthetic runs and repeatability;
  - grouped-fold and class-completeness audits;
  - no validation/test contribution to normalization, positive weights, epoch selection, or
    optimization;
  - finite loss, gradients, probabilities, predictions, and seed diagnostics.
- Completion criteria: every primary architecture/preprocessing variant satisfies the shared
  random-imagery prediction contract without leakage.
- Review gate: Stop and wait for explicit user approval.

### 4. Artifacts, Workflow, And CLI - Completed

- Objective: Add immutable Torch artifacts and complete terminal workflows.
- Deliverables:
  - Torch artifact schema under
    `artifacts/experiments/random-imagery-torch/<model-id>/<config-hash>/`;
  - atomic writer, safe reader, and trusted replay;
  - shared train-or-reuse workflow;
  - `random-imagery-torch run`, `evaluate`, and `compare` plus module entry point.
- Constraints:
  - persist three `state_dict` files, preprocessing identity, normalization state, histories,
    split/leakage audits, per-seed and ensemble outputs, metrics, environment, and CUDA metadata;
  - inventory every file with SHA-256 and byte size;
  - safe mode must never load model weights;
  - trusted replay may call `torch.load(..., weights_only=True)` only after full manifest,
    configuration, input-axis, channel, and sample-key validation;
  - CPU loading of CUDA-trained artifacts must remain supported;
  - runs are immutable and incomplete reuse is rejected.
- Verification:
  - synthetic round trip and exact replay;
  - corruption, unexpected file, unsafe path, wrong axis, wrong split, duplicate, and incomplete
    reuse rejection;
  - CLI parity and compatible classical/Torch comparison.
- Completion criteria: every primary variant can be trained, safely evaluated, replayed, and
  immutably reused from the terminal.
- Review gate: Stop and wait for explicit user approval.

### 5. Full Real-Corpus Training - Completed

- Objective: Execute all 12 primary Torch variants under both established protocols.
- Deliverables:
  - executed `notebooks/6.0-torch-spectral-models-training.ipynb`;
  - 36 validated immutable direction runs;
  - training curves, preprocessing shapes, parameter counts, selected epochs, seed variation,
    GPU runtime, and leakage summaries.
- Constraints:
  - notebook calls reusable workflow code;
  - every second execution must use immutable reuse;
  - cross-subject directions retain 141/39 rows;
  - each cross-trial direction retains 81/81 rows;
  - training uses the available CUDA device while artifacts remain CPU-loadable.
- Verification:
  - all expected model/protocol/direction runs exist exactly once;
  - finite gradients, scores, metrics, histories, and ensemble predictions;
  - clean leakage audits and expected three checkpoints per direction;
  - executed notebook marker, no cell errors, integration test, and visual figure inspection.
- Completion criteria: the complete source-backed Torch result set is available for final
  comparison.
- Review gate: Stop and wait for explicit user approval.

### 6. Final Comparison - Awaiting Review

- Objective: Compare Torch, classical, and non-EEG models without conflating protocols or score
  semantics.
- Deliverables:
  - executed `notebooks/6.1-torch-classical-comparison.ipynb`;
  - separate cross-subject and combined cross-trial tables and figures;
  - balanced accuracy, Brier/score-MSE, IoU, Hamming loss, exact match, pooled ECE, runtime,
    parameter count, and seed-sensitivity diagnostics;
  - paired subject-bootstrap contrasts against Logistic Regression.
- Constraints:
  - require exact ordered sample keys, targets, and subject IDs;
  - use the same 2,000 subject-cluster bootstrap draws within each protocol;
  - report pointwise intervals and Holm-adjusted bootstrap p-values for the 21 non-reference
    learned models;
  - do not average protocols;
  - do not promote descriptive ranks to superiority claims without multiplicity-adjusted evidence.
- Verification:
  - reproducible tables and bootstrap draws;
  - calibration and seed diagnostics use only compatible probability scores;
  - notebook integration test, visual inspection, Ruff, lockfile check, full pytest, and diff
    check.
- Completion criteria: the Torch study is reproducible, uncertainty-aware, multiplicity-qualified,
  and directly comparable to the established classical study.
- Review gate: Stop and wait for explicit user approval.

## Decisions And Assumptions

- Primary model IDs are the Cartesian product of `eegnet`, `deep-convnet`, and
  `shallow-convnet` with `fft`, `morlet`, `superlet`, and `stft`, using a `-multilabel` suffix.
- All 12 architecture/preprocessing combinations receive outer-test evaluation; preprocessing is
  not selected by outer-train validation.
- One model jointly predicts all 36 pixels through independent sigmoid logits.
- Three final seeds form one ensemble result and are not counted as separate outer-test models.
- Input power uses a stable logarithm followed by train-only z-scoring per frequency bin.
- No data augmentation, pretrained weights, AMP, post-hoc calibration, or hyperparameter grid is
  included.
- TensorFlow and PyRiemann are not dependencies of the new training path.
- The implementations derive from ARL EEGModels:
  `https://github.com/vlawhern/arl-eegmodels`.
- Preserve the upstream CC0/Apache-2.0 license text from:
  `https://raw.githubusercontent.com/vlawhern/arl-eegmodels/master/LICENSE.txt`.
- Approval of this plan does not alter the review state of the separate classical-model plan.

## Progress Log

- 2026-06-15: User requested a staged PyTorch study derived from `eegnet-tesnorflow.py`, with the
  supplied file removed after migration.
- 2026-06-15: Repository inspection found five supplied architectures, an existing Torch spectral
  dataset API, four established preprocessing methods, the canonical 180-row/36-target dataset,
  and an available NVIDIA GeForce RTX 3070 Ti.
- 2026-06-15: User selected three primary architectures, one 36-output multi-label network,
  full factorial evaluation across four spectral methods, frequency-plane adapters, three-seed
  ensembles, fixed training defaults, log plus train-only z-score normalization, and comparison
  with all classical models and baselines.
- 2026-06-15: User explicitly requested that the plan be saved. Plan approved and persisted;
  implementation has not started.
- 2026-06-15: User explicitly requested implementation to begin. Stage 1 marked In Progress.
- 2026-06-15: Stage 1 implemented in `experiments/random_imagery_torch/` with a distinct
  crop-before-transform cache, immutable spectral and normalization contracts, train-key-only
  log-power frequency normalization, exact target alignment, lazy test access, and Torch collation.
- 2026-06-15: Canonical random-imagery key `(1, 1, 7)` produced the expected real shapes:
  FFT `(63, 39)`, Morlet `(63, 39, 49)`, Superlet `(63, 39, 46)`, and STFT `(63, 39, 51)`.
  Time-frequency axes are stored in source-recording seconds after the `[0.5, 15.5)` crop.
- 2026-06-15: A real cross-subject smoke check fitted normalization only on train key
  `(1, 1, 7)` and materialized test key `(9, 1, 7)`. Batch shapes were `(1, 1, 63, 39)` for FFT
  and `(1, 39, 63, 49/46/51)` for Morlet/Superlet/STFT, with aligned `(1, 36)` targets.
- 2026-06-15: Verification passed: 10 focused Stage 1 tests, 56 focused spectral/Torch regression
  tests, `uv run ruff check .`, `uv lock --check`, `git diff --check`, and the full suite with
  344 passed. Two pre-existing Python 3.13 multiprocessing `fork()` deprecation warnings remain.
- 2026-06-15: Stage 1 marked Awaiting Review. Stage 2 has not started.
- 2026-06-15: User approved Stage 1 by requesting continuation. Stage 1 marked Completed and
  Stage 2 marked In Progress.
- 2026-06-15: Stage 2 added typed PyTorch spectral adaptations of EEGNet, DeepConvNet,
  ShallowConvNet, EEGNet-SSVEP, and EEGNet-v1. Every model consumes an exact declared
  `(planes, electrodes, width)` shape and returns 36 logits without an embedded sigmoid.
- 2026-06-15: The ports preserve source filter counts, EEGNet depthwise/separable structure,
  nonlinearities, dropout families, and max-norm intent. TensorFlow-style SAME padding supports
  even and strided kernels; kernels are capped at available spectral width and adaptive global
  pooling replaces incompatible fixed flatten geometries.
- 2026-06-15: Added explicit max-norm projection for constrained Conv2d and Linear output filters,
  deterministic Xavier initialization, architecture factory/registry constants, strict input
  validation, parameter counts, and primary versus exploratory architecture groups.
- 2026-06-15: Retained the complete upstream CC0 1.0/Apache-2.0 license and a modification/citation
  notice inside `experiments/random_imagery_torch/`; setuptools wheel verification confirmed both
  files are packaged. Removed the supplied `eegnet-tesnorflow.py` after all port tests passed.
- 2026-06-15: Added 69 architecture tests. All five models pass finite CPU forward/backward on all
  four input geometries, deterministic initialization, depthwise grouping, max-norm, exact output,
  parameter snapshot, license, and CUDA smoke checks.
- 2026-06-15: A real canonical CUDA smoke check passed for all 12 primary
  architecture/preprocessing combinations on the RTX 3070 Ti, including backward and max-norm
  projection. Parameter counts range from 2,412-18,060 for EEGNet, 179,136-183,886 for
  DeepConvNet, and 102,916-122,676 for ShallowConvNet across the four representations.
- 2026-06-15: Verification passed: 79 focused Torch input/model tests, `uv run ruff check .`,
  `uv lock --check`, wheel/sdist build, `git diff --check`, and the full suite with 413 passed.
  Two pre-existing Python 3.13 multiprocessing `fork()` deprecation warnings remain.
- 2026-06-15: Stage 2 marked Awaiting Review. Stage 3 has not started.
- 2026-06-15: User approved Stage 2 by requesting continuation. Stage 2 marked Completed and
  Stage 3 marked In Progress.
- 2026-06-15: Stage 3 added strict `TorchTrainingConfig`, grouped three-fold subject validation,
  fold-local train-only normalization and positive class weights, finite BCE/gradient checks,
  max-norm projection after optimizer updates, early stopping, median fold-best epoch selection,
  and final three-seed ensemble fitting.
- 2026-06-15: Stage 3 split fitting and prediction into explicit calls. Fitting validates the
  evaluation direction and materializes only direction-training and fold-validation rows; outer
  test rows are accepted only by `predict_torch_ensemble(...)` after all final seed checkpoints
  exist.
- 2026-06-15: Added immutable training schemas for grouped folds, validation histories, CPU
  checkpoint snapshots, epoch-selection diagnostics, ensemble members, fitted ensembles, and
  mean-probability predictions compatible with the shared random-imagery prediction contract.
- 2026-06-15: Added 7 focused training tests covering subject-disjoint folds, class completeness,
  train-only positive weights, no outer-test access during fitting, fold-normalization provenance,
  deterministic repeatability, ensemble score averaging, missing-class rejection, and CUDA smoke
  when available.
- 2026-06-15: Verification passed: 7 Stage 3 tests, 86 focused Torch input/model/training tests,
  `uv run ruff check .`, `uv lock --check`, `git diff --check`, and the full suite with
  420 passed. Two pre-existing Python 3.13 multiprocessing `fork()` deprecation warnings remain.
- 2026-06-15: Stage 3 marked Awaiting Review. Stage 4 has not started.
- 2026-06-15: User approved Stage 3 by requesting continuation. Stage 3 marked Completed and
  Stage 4 marked In Progress.
- 2026-06-15: Stage 4 added `TorchExperimentConfig`, the primary 12-model ID parser, default
  `confs/experiments/random_imagery_torch.yaml`, versioned run hashing, and the
  `random-imagery-torch` console/module entry point.
- 2026-06-15: Stage 4 added immutable Torch direction artifacts under
  `artifacts/experiments/random-imagery-torch/<model-id>/<config-hash>/`, including config,
  environment, split, preprocessing identity, normalization state, training histories, three
  state-dict checkpoints, member/ensemble outputs, metrics, baselines, and SHA-256 inventory.
- 2026-06-15: Safe loading validates manifests, hashes, arrays, metrics, bootstrap summaries,
  checkpoint metadata, and unsafe paths without calling `torch.load`. Trusted replay requires
  `load_torch_run(..., trusted=True)` and uses `torch.load(..., weights_only=True)` after
  validation.
- 2026-06-15: Added `execute_torch_protocol(...)` train-or-reuse workflow. Reuse requires every
  expected direction run to exist, validate, match the exact resolved config, and match the
  spectral input config hash; duplicate training without reuse is rejected as immutable.
- 2026-06-15: Added 8 focused Stage 4 tests for artifact round trip, safe load, trusted replay,
  tamper rejection, unsafe checkpoint rejection, workflow reuse, immutable duplicate rejection,
  CLI parsing, JSON output, failure codes, and console/module entry point parity.
- 2026-06-15: Verification passed: 94 focused Torch input/model/training/artifact/workflow/CLI
  tests, `uv run ruff check .`, `uv lock --check`, `git diff --check`, and the full suite with
  428 passed. Two pre-existing Python 3.13 multiprocessing `fork()` deprecation warnings remain.
- 2026-06-15: Stage 4 marked Awaiting Review. Stage 5 has not started.
- 2026-06-16: User approved Stage 4 by requesting continuation. Stage 4 marked Completed and
  Stage 5 marked In Progress.
- 2026-06-16: Stage 5 added and executed
  `notebooks/6.0-torch-spectral-models-training.ipynb`, which ran all 12 primary Torch
  architecture/preprocessing variants across the cross-subject and bidirectional cross-trial
  protocols through `execute_torch_protocol(...)`.
- 2026-06-16: Published 36 validated immutable Torch direction runs under
  `artifacts/experiments/random-imagery-torch/`: 12 cross-subject, 12 trial-1-to-trial-2, and
  12 trial-2-to-trial-1. The notebook's second pass verified immutable reuse for all 36 runs.
- 2026-06-16: The run environment recorded CUDA on `NVIDIA GeForce RTX 3070 Ti`. Crop-spectral
  caches for FFT, Morlet, Superlet, and STFT were populated under
  `artifacts/preprocessed-imagery/Data_Pattern/patt/`.
- 2026-06-16: Cross-subject direction balanced accuracy ranged from 0.486743 to 0.513443
  (mean 0.500453). Within-subject direction balanced accuracy ranged from 0.479567 to 0.524497
  (mean 0.502307). Combined within-subject descriptive leader was
  `deep-convnet-stft-multilabel` with balanced accuracy 0.512011 and 95% subject-bootstrap
  interval [0.500668, 0.520872].
- 2026-06-16: Visual inspection of the two notebook figures passed. Verification passed:
  97 focused Torch/notebook tests, `uv run ruff check .`, `uv lock --check`,
  `git diff --check`, and the full suite with 429 passed. Two pre-existing Python 3.13
  multiprocessing `fork()` deprecation warnings remain.
- 2026-06-16: Stage 5 marked Awaiting Review. Stage 6 has not started.
- 2026-06-16: User approved Stage 5 by requesting continuation. Stage 5 marked Completed and
  Stage 6 marked In Progress.
- 2026-06-16: Stage 6 added and executed
  `notebooks/6.1-torch-classical-comparison.ipynb`, comparing Logistic Regression, nine
  classical schema-v3 variants, 12 Torch spectral variants, and canonical non-EEG baselines from
  immutable artifacts without loading joblib pipelines or Torch checkpoint weights.
- 2026-06-16: The final comparison required exact ordered test sample keys, targets, and subject
  IDs against Logistic Regression for every model. Cross-subject comparison used 39 held-out rows
  from seven subjects; combined bidirectional cross-trial comparison used 162 held-out rows from
  27 identities.
- 2026-06-16: The notebook used the same 2,000 subject-cluster bootstrap draws within each
  protocol and reported Holm-adjusted bootstrap p-values across the 21 non-reference learned
  models. Minimum Holm-adjusted balanced-accuracy p-value was 0.273000.
- 2026-06-16: Cross-subject descriptive leader was
  `ridge-regression-independent` at balanced accuracy 0.518382. Combined within-subject
  descriptive leader was `deep-convnet-stft-multilabel` at balanced accuracy 0.512011. No model
  is promoted as superior to Logistic Regression under the multiplicity-aware paired bootstrap
  screen.
- 2026-06-16: Visual inspection of all six final-comparison figures passed. Verification passed:
  10 focused comparison/notebook tests, `uv run ruff check .`, `uv lock --check`,
  `git diff --check`, and the full suite with 430 passed. Two pre-existing Python 3.13
  multiprocessing `fork()` deprecation warnings remain.
- 2026-06-16: Stage 6 marked Awaiting Review. Because this is the final stage, the plan can be
  marked completed after explicit user approval.
