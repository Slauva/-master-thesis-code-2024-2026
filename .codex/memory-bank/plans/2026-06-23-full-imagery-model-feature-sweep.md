# Full Imagery Model Feature Sweep

Status: completed
Last updated: 2026-06-24
Next stage: complete

## Goal

Run the project imagery reconstruction task on the full `Data_Pattern/patt` corpus, including both
`geometric` and `random` samples, across every available model and feature-family combination with
leakage-aware protocols, immutable artifacts, and a final reproducible comparison notebook.

## Scope

- Use the full local `Data_Pattern/patt` dataset: 540 blocks total, 360 `geometric` and 180
  `random`, spanning 33 subjects and 60 subject-trial pairs.
- Preserve the 6x6 binary image reconstruction target using the row-major image payload from both
  sample types.
- Evaluate the established protocols separately:
  - cross-subject subject-disjoint split;
  - bidirectional within-subject cross-trial split.
- Run fixed feature-family combinations for tabular models:
  - `time`;
  - `spectral`;
  - `time+spectral`;
  - `covariance`;
  - `correlation`;
  - `log_covariance`;
  - `lndp`;
  - `lgp`;
  - `lbp`.
- Run the available classical/common random-imagery models:
  - `logistic-regression-independent`;
  - `linear-svm-independent`;
  - `ridge-classifier-independent`;
  - `ridge-regression-independent`;
  - `ridge-regression-multioutput`;
  - `elastic-net-independent`;
  - `elastic-net-multioutput`;
  - `random-forest-independent`;
  - `random-forest-multioutput`;
  - `pls-regression-multioutput`.
- Run the available Torch spectral primary combinations:
  - architectures: `eegnet`, `deep-convnet`, `shallow-convnet`;
  - methods: `fft`, `morlet`, `superlet`, `stft`.
- Keep raw FIF files unchanged and write derived outputs only under generated artifact locations.

## Exclusions

- Do not mix `Data_Train/exec` into this sweep unless a later explicitly approved plan revision
  changes the task definition.
- Do not tune thresholds on held-out test data; keep the fixed `0.5` decision threshold.
- Do not average cross-subject and within-subject protocols into one headline score.
- Do not claim model superiority from descriptive ranks without paired uncertainty checks.

## Acceptance Criteria

- Full-dataset target construction accepts both `GeometricSample` and `RandomSample` without
  weakening image-shape, binary-target, sample-key, or fingerprint validation.
- Cross-subject and within-subject protocol audits pass with no forbidden leakage.
- Tabular model runs represent true fixed `model x feature_family` combinations, not hidden
  train-only selection over multiple feature families.
- Immutable artifacts can be safely reloaded or reused without trusting joblib by default.
- The final notebook/report presents protocol-separated metrics, uncertainty, paired comparisons,
  and clear caveats suitable for thesis discussion.
- Verification commands and notebook execution evidence are recorded before each stage is offered
  for review.

## Major Risks

- The current random-imagery contracts only accept `pattern_type="random"` and
  `RandomSample` targets; Stage 1 must fix this before any full-corpus training.
- Existing model workflows select one feature family from candidates; the sweep must force one
  candidate per run to produce a real `model x feature_family` matrix.
- The full sweep is much larger than previous random-only runs: tabular runs cover 180 protocol
  combinations and up to 270 direction artifacts, and Torch adds 24 protocol combinations and up
  to 36 direction artifacts.
- Some combinations may be slow or numerically fragile, especially ElasticNet and Torch
  time-frequency models; failures must be explicit artifacts or stage blockers, not silently
  omitted.

## Stages

### 1. Full Dataset Contract And Leakage Audit - Completed

- Objective: Extend and validate the dataset/target/protocol contract for `geometric+random`
  `Data_Pattern/patt` rows before any model training.
- Deliverables:
  - updated dataset selection and target-building code;
  - focused tests for mixed `GeometricSample` and `RandomSample` targets;
  - a small audit artifact under `artifacts/experiments/full-imagery/`;
  - documented counts for total rows, sample types, subjects, trials, split rows, and per-pixel
    class supports.
- Constraints:
  - preserve canonical sample keys `(subject_id, trial_number, block_index)`;
  - keep subject-disjoint cross-subject evaluation for generalization claims;
  - allow subject overlap only in the existing cross-trial protocol while keeping sample keys,
    trial numbers, random seeds, and random image fingerprints disjoint per direction;
  - permit deterministic geometric pattern repeats only as tracked task-label provenance;
  - never use `sample_type`, `pattern_id`, or `seed` as EEG features.
- Verification:
  - focused dataset/protocol tests;
  - full-dataset index audit without loading all FIF arrays;
  - `uv run ruff check .`;
  - `uv run pytest` if runtime is reasonable for this structural change.
- Completion criteria:
  - mixed full-corpus target construction passes validation;
  - both evaluation protocols pass leakage and class-support checks;
  - old random-only tests remain compatible or are intentionally updated with coverage.
- Review gate: Stop and wait for explicit user approval.

Stage 1 result on 2026-06-23:

- Delivered code changes:
  - `DatasetSelectionConfig.pattern_type` now accepts `geometric`, `random`, or `None`;
  - `DatasetSelectionConfig.target_sample_types` maps `None` to `("geometric", "random")`;
  - `build_random_imagery_targets(...)` can explicitly accept mixed geometric/random samples while
    preserving the old random-only default;
  - random seed leakage checks ignore the `-1` sentinel used for samples without random seeds;
  - workflow/runner entry points pass the configured allowed target sample types;
  - `experiments/random_imagery/full_dataset_audit.py` writes a metadata-only full-corpus audit.
- Artifact:
  - `artifacts/experiments/full-imagery/stage1_full_dataset_audit.json`.
- Audit result:
  - mixed full-corpus targets build successfully with shape `(540, 36)`;
  - the full corpus contains 360 `geometric` rows and 180 `random` rows;
  - cross-subject split rows are 423 train and 117 test from 26 and 7 subjects;
  - within-subject directions are 243/243 rows each over 27 eligible subjects;
  - all pixel tasks have both classes in train and test;
  - no sample-key or random-seed overlap was found.
- Blocker:
  - the approved image-payload disjointness contract fails for the full corpus;
  - cross-subject and both within-subject directions each contain 13 overlapping image
    fingerprints between train and test;
  - these overlaps come from the 13 deterministic geometric patterns, which repeat across
    subjects/trials by dataset design.
- Verification:
  - `uv run pytest tests/experiments/test_logistic_regression_config.py
    tests/experiments/test_logistic_regression_data.py` passed with 29 tests;
  - `uv run ruff check .` passed;
  - full `uv run pytest` passed with 481 tests and 2 pre-existing multiprocessing warnings.
- Required decision:
  - either revise the full-imagery leakage contract to permit deterministic geometric pattern
    repeats as task labels while still forbidding random seed/sample-key leakage;
  - or keep the current image-payload disjointness contract, which blocks any full-dataset run that
    includes repeated geometric patterns.

Stage 1 resolution on 2026-06-23:

- User's "дальше" was treated as approval to revise the leakage contract.
- Deterministic `geometric` pattern repeats are allowed as repeated task labels and tracked through
  `overlapping_geometric_pattern_ids`.
- Random image fingerprint overlaps remain forbidden through
  `overlapping_random_image_fingerprints`, alongside sample-key and random-seed leakage.
- Regenerated audit artifact reports `stage1_status="ready"`, no forbidden leakage, zero random
  image overlaps, and 13 repeated geometric pattern IDs `0`-`12` as expected.

### 2. Fixed Feature Matrix Runner - Completed

- Objective: Add a matrix-run path that treats each feature family as a fixed experimental
  condition instead of selecting among multiple candidates inside one run.
- Deliverables:
  - configuration or CLI support for fixed feature-family sweeps;
  - Logistic Regression integration into the same sweep accounting as the schema-v3 models, or an
    explicitly compatible reference bridge;
  - tests proving each run materializes and trains on exactly the requested feature family.
- Constraints:
  - fit variance filters, selectors, scalers, calibrators, and all model parameters only inside
    training folds or final training rows;
  - materialize held-out features only after model selection and fitting are complete;
  - keep artifact hashes sensitive to model ID, feature family, protocol, direction, and full
    dataset contract.
- Verification:
  - focused runner/config/CLI tests;
  - synthetic full-matrix smoke that covers representative classifier, regressor, and Logistic
    Regression paths;
  - `uv run ruff check .`;
  - `uv run pytest`.
- Completion criteria:
  - a single command or workflow can enumerate planned tabular runs without ambiguity;
  - duplicate immutable runs are refused unless `reuse_existing` validates the complete expected
    set.
- Review gate: Stop and wait for explicit user approval.

Stage 2 result on 2026-06-23:

- Delivered code changes:
  - `experiments/random_imagery/matrix.py` defines the fixed classical matrix contract;
  - `random-imagery-models matrix-plan` enumerates run commands without executing training;
  - the matrix includes the Logistic Regression reference bridge plus all nine schema-v3 classical
    models;
  - every planned run sets `dataset.pattern_type=null`, so full `geometric+random` targets are used;
  - every planned run sets exactly one `feature_screening.candidates` entry, so the run is a fixed
    feature-family condition rather than train-side selection over many families;
  - artifact roots are partitioned by `model_id/feature_slug` under
    `artifacts/experiments/full-imagery/classical/`.
- Matrix coverage:
  - 10 model IDs x 9 feature families x 2 protocols = 180 protocol specs;
  - cross-subject contributes one direction each and within-subject contributes two directions
    each, for 270 expected direction runs;
  - feature families are `time`, `spectral`, `time+spectral`, `covariance`, `correlation`,
    `log_covariance`, `lndp`, `lgp`, and `lbp`.
- CLI evidence:
  - `uv run python -m experiments.random_imagery matrix-plan --json` reported
    `run_count=180` and `expected_direction_run_count=270`;
  - a representative emitted command is
    `logistic-regression run --protocol cross-subject --set dataset.pattern_type=null --set
    feature_screening.candidates=[[time,spectral]] --set
    artifacts.root=artifacts/experiments/full-imagery/classical/logistic-regression-independent/time+spectral`.
- Tests:
  - `tests/experiments/test_random_imagery_matrix.py` validates enumeration, runner selection,
    plan IDs, config override loading, full-dataset target selection, and fixed single-candidate
    feature families;
  - `tests/experiments/test_random_imagery_cli.py` validates the `matrix-plan` CLI and confirms it
    does not call training;
  - `tests/experiments/test_random_imagery_framework.py` now includes a synthetic runner check that
    a single-candidate config trains and predicts with exactly the requested feature family.
- Verification:
  - `uv run pytest tests/experiments/test_random_imagery_matrix.py
    tests/experiments/test_random_imagery_cli.py
    tests/experiments/test_random_imagery_framework.py` passed with 26 tests;
  - `uv run ruff check experiments/random_imagery/matrix.py experiments/random_imagery/cli.py
    experiments/random_imagery/__init__.py tests/experiments/test_random_imagery_matrix.py
    tests/experiments/test_random_imagery_cli.py tests/experiments/test_random_imagery_framework.py`
    passed;
  - `uv run ruff check .` passed;
  - full `uv run pytest` passed with 498 tests and 2 pre-existing multiprocessing warnings.
- Limitation:
  - Stage 2 enumerates and validates run specifications only; Stage 3 is still responsible for
    executing the 180 protocol runs, validating immutable artifacts, and logging any slow or failed
    combinations.

### 3. Full Classical Model Feature Sweep - Completed

- Objective: Execute all tabular `model x feature_family x protocol` combinations on the full
  corpus and persist immutable artifacts.
- Deliverables:
  - immutable run directories under a new full-imagery artifact root;
  - safe summary JSON covering every completed direction;
  - failure log for any blocked model/feature/protocol combination, if any.
- Constraints:
  - keep cross-subject and within-subject outputs separate;
  - keep regression clipped scores semantically separate from classifier probabilities;
  - do not silently drop slow or failed combinations.
- Verification:
  - safe artifact validation for every run directory;
  - reuse pass over the complete expected run set;
  - `uv run ruff check .`;
  - focused artifact/workflow tests, plus full `uv run pytest` if feasible.
- Completion criteria:
  - every planned classical combination is either completed and validated or explicitly blocked with
    a reproducible reason;
  - run summaries include primary metrics, baselines, score diagnostics, selected hyperparameters,
    and bootstrap intervals.
- Review gate: Stop and wait for explicit user approval.

Stage 3 result on 2026-06-23:

- Delivered code changes:
  - `experiments/random_imagery/matrix.py` now executes the fixed classical matrix through
    `execute_classical_matrix_sweep(...)`, reusing validated existing runs when possible;
  - `random-imagery-models matrix-run` runs the full or filtered matrix, persists an incremental
    summary JSON, and writes an explicit failure log;
  - non-fatal `scipy.linalg.LinAlgWarning` noise from numerically difficult Ridge fits is
    suppressed only around individual model execution so the sweep log remains readable.
- Artifacts:
  - summary: `artifacts/experiments/full-imagery/stage3_classical_matrix_summary.json`;
  - failure log: `artifacts/experiments/full-imagery/stage3_classical_matrix_failures.json`;
  - immutable run roots:
    `artifacts/experiments/full-imagery/classical/<model_id>/<feature_slug>/`.
- Execution result:
  - planned coverage remained 180 protocol specs and 270 expected direction runs;
  - 167 protocol specs completed;
  - 249 direction run directories were completed and safely reloaded;
  - 13 protocol specs failed explicitly with convergence warnings and are recorded in the failure
    log rather than omitted;
  - validated schema counts were 27 schema-v2 Logistic Regression run directories and 222
    schema-v3 model run directories.
- Failure summary:
  - `linear-svm-independent`: 5 convergence failures on `time` cross/within, `covariance`
    cross/within, and `correlation` within;
  - `elastic-net-independent`: 6 convergence failures on `time` cross/within, `spectral` within,
    `time+spectral` within, and `covariance` cross/within;
  - `elastic-net-multioutput`: 2 convergence failures on `covariance` cross/within.
- Verification:
  - safe artifact reload validated all 249 completed direction run directories;
  - `uv run ruff check .` passed;
  - `uv run pytest` passed with 501 tests and 2 pre-existing multiprocessing warnings;
  - `git diff --check` passed.
- Limitation:
  - `matrix-run` exits with status code 1 when any protocol specs fail, even after completing the
    sweep, so convergence-blocked combinations stay visible. Stage 3 therefore satisfies the
    completion criterion as completed-or-explicitly-blocked, pending user review.

### 4. Full Torch Spectral Sweep - Completed

- Objective: Execute primary Torch spectral architecture/method combinations on the same full
  corpus protocols.
- Deliverables:
  - immutable Torch run directories for `eegnet`, `deep-convnet`, and `shallow-convnet` crossed
    with `fft`, `morlet`, `superlet`, and `stft`;
  - safe summary JSON and reuse validation;
  - runtime and resource notes.
- Constraints:
  - build Torch spectral inputs from the exact `[0.5, 15.5)` imagery crop;
  - fit spectral normalization and positive weights only from training rows;
  - keep fitting and test prediction as separate leakage boundaries;
  - keep ensemble seeds and thresholds fixed by configuration.
- Verification:
  - focused Torch workflow/artifact tests;
  - safe validation for every Torch run directory;
  - reuse pass over the complete expected run set;
  - `uv run ruff check .`;
  - `uv run pytest` if feasible after long-running training.
- Completion criteria:
  - every planned Torch combination is completed and validated or explicitly blocked with a
    reproducible reason.
- Review gate: Stop and wait for explicit user approval.

Stage 4 result on 2026-06-23:

- Delivered code changes:
  - `experiments/random_imagery_torch/matrix.py` defines the full-imagery Torch matrix contract and
    executor;
  - `random-imagery-torch matrix-plan` enumerates the Torch architecture/method/protocol grid;
  - `random-imagery-torch matrix-run` executes or reuses the matrix, writes an incremental summary,
    and persists a failure log;
  - `CropSpectralDataset` and `CropSpectralSample` now accept the full mixed
    `GeometricSample | RandomSample` corpus instead of requiring random-only metadata.
- Artifacts:
  - summary: `artifacts/experiments/full-imagery/stage4_torch_matrix_summary.json`;
  - failure log: `artifacts/experiments/full-imagery/stage4_torch_matrix_failures.json`;
  - immutable run roots: `artifacts/experiments/full-imagery/torch/<model_id>/<run_hash>/`;
  - derived spectral cache entries under `artifacts/preprocessed-imagery/Data_Pattern/patt/`.
- Execution result:
  - planned coverage was 12 Torch model IDs x 2 protocols = 24 protocol specs;
  - all 24 protocol specs completed;
  - all 36 expected direction run directories were completed and safely reloaded;
  - failure count was zero;
  - reuse pass over the complete expected run set succeeded with all specs reused.
- Runtime notes:
  - immutable `training.json` files report 36 direction fits, total training time about
    5114.05 seconds;
  - per-direction training seconds ranged from 26.42 to 832.04 with median 123.86;
  - the first `eegnet-superlet` cross-subject protocol was the slowest protocol wall-time segment
    because it had to materialize much of the full-corpus Superlet spectral cache.
- Descriptive metric snapshot:
  - best cross-subject mean balanced accuracy in the Stage 4 summary is
    `eegnet-stft-multilabel` at `0.525845`;
  - next cross-subject leaders are `eegnet-superlet-multilabel` at `0.516823` and
    `eegnet-morlet-multilabel` at `0.513022`;
  - best within-subject combined mean balanced accuracy in the Stage 4 summary is
    `deep-convnet-superlet-multilabel` at `0.509403`.
- Verification:
  - focused Torch matrix/input/workflow tests passed with 18 tests;
  - safe artifact reload validated all 36 completed direction run directories;
  - `uv run python -m experiments.random_imagery_torch matrix-run --json` succeeded as a reuse
    pass with 24 completed protocol specs, 36 direction runs, and zero failures;
  - `uv run ruff check .` passed;
  - full `uv run pytest` passed with 508 tests and 2 pre-existing multiprocessing warnings;
  - `git diff --check` passed.
- Limitation:
  - the current summary file reflects the final reuse-validation pass, so per-spec matrix
    `duration_seconds` values are reuse validation durations. Actual training times remain in each
    immutable run's `training.json`.

### 5. Comparison Notebook And Report - Completed

- Objective: Build a reproducible notebook/report that compares the full sweep results without
  overstating weak or near-chance effects.
- Deliverables:
  - executed notebook, proposed path:
    `notebooks/6.2-full-imagery-model-feature-sweep.ipynb`;
  - summary JSON under `artifacts/experiments/full-imagery/`;
  - protocol-separated tables and figures for model, feature family, metrics, uncertainty, and
    paired differences.
- Constraints:
  - compare only runs with exactly matching ordered test sample keys, targets, and subject IDs;
  - use paired subject-cluster bootstrap draws within protocol;
  - treat paired intervals as exploratory pointwise intervals unless a multiplicity strategy is
    explicitly added;
  - never average cross-subject and within-subject protocols.
- Verification:
  - notebook executes top-to-bottom with a clear verification marker and no stored error outputs;
  - summary JSON validates expected model/feature/protocol coverage;
  - `uv run ruff check .`;
  - notebook validation tests or focused static checks.
- Completion criteria:
  - final report identifies descriptive leaders, uncertainty, baselines, and limitations;
  - all claims are traceable to immutable artifacts.
- Review gate: Stop and wait for explicit user approval.

Stage 5 result on 2026-06-24:

- Delivered code and report changes:
  - `experiments/random_imagery/full_sweep_comparison.py` safely loads the immutable Stage 3 and
    Stage 4 run arrays, validates ordered test sample keys/targets/subject IDs, and computes
    protocol-separated paired subject-cluster bootstrap summaries;
  - `notebooks/6.2-full-imagery-model-feature-sweep.ipynb` executes top-to-bottom, writes the
    Stage 5 JSON summary, renders protocol-separated tables/figures, and prints
    `FULL_IMAGERY_SWEEP_COMPARISON_VERIFIED`;
  - tests validate the bootstrap helper and the executed notebook contract.
- Artifacts:
  - summary: `artifacts/experiments/full-imagery/stage5_comparison_summary.json`;
  - figures under `artifacts/experiments/full-imagery/stage5_figures/`:
    `cross_subject_top_learned.png`, `within_subject_top_learned.png`,
    `cross_subject_paired_delta_vs_logreg_lbp.png`,
    `within_subject_paired_delta_vs_logreg_lbp.png`, `feature_method_family_maxima.png`, and
    `reference_baselines.png`.
- Coverage:
  - 191 completed learned protocol conditions were compared: 97 cross-subject and 94
    within-subject;
  - completed direction runs total 285: 249 classical plus 36 Torch;
  - all completed conditions were paired-compatible with the protocol reference split;
  - the 13 classical convergence failures remain explicit carried-forward failures.
- Descriptive leaders:
  - cross-subject top learned condition is `torch:eegnet-stft-multilabel:stft:cross-subject`,
    mean balanced accuracy `0.525845`, pointwise 95% subject-cluster interval
    `[0.497681, 0.556626]`;
  - paired delta versus `classical:logistic-regression-independent:lbp:cross-subject` is
    `+0.036915`, pointwise interval `[+0.005357, +0.067819]`;
  - within-subject top learned condition is
    `classical:pls-regression-multioutput:lgp:within-subject`, mean balanced accuracy `0.511556`,
    pointwise 95% subject-cluster interval `[0.501478, 0.521404]`;
  - paired delta versus `classical:logistic-regression-independent:lbp:within-subject` is
    `+0.015269`, pointwise interval `[+0.001203, +0.028341]`.
- Caveat:
  - paired intervals are exploratory pointwise intervals and are not multiplicity-adjusted;
  - cross-subject and within-subject scores are not averaged together;
  - near-chance balanced accuracy remains the core interpretation constraint.
- Verification:
  - notebook execution via `jupyter nbconvert --execute --to notebook --inplace
    notebooks/6.2-full-imagery-model-feature-sweep.ipynb` passed after using approved elevated
    execution for Jupyter kernel sockets;
  - focused Stage 5 tests passed with 4 tests;
  - `uv run ruff check .` passed;
  - full `uv run pytest` passed with 512 tests and 2 pre-existing multiprocessing warnings;
  - `git diff --check` passed.

### 6. Final Memory Update And Handoff - Completed

- Objective: Close the staged plan with durable memory notes and a concise handoff for thesis use.
- Deliverables:
  - updated `.codex/memory-bank/active_context.md`;
  - updated `.codex/memory-bank/experiments.md`;
  - durable decisions added to `.codex/memory-bank/decisions.md` only if new scientific or
    architectural choices were accepted during execution.
- Constraints:
  - do not store raw sensitive labels beyond aggregate counts and artifact paths;
  - preserve caveats about population, leakage boundaries, and near-chance results.
- Verification:
  - `git diff --check`;
  - static checks for conflict markers and obvious placeholders in touched memory files.
- Completion criteria:
  - plan status is set to `completed`;
  - final artifact paths, verification evidence, and interpretation caveats are easy to recover.
- Review gate: Stop and wait for explicit user approval.

Stage 6 result on 2026-06-24:

- User's "дальше" was treated as approval of Stage 5 and authorization to run the final handoff
  stage.
- Marked this staged plan as `completed` and set `Next stage: complete`.
- Updated durable memory:
  - `.codex/memory-bank/active_context.md` records the full-imagery plan as completed and preserves
    the thesis-use handoff;
  - `.codex/memory-bank/experiments.md` already records the Stage 5 reproducible comparison
    artifact, coverage, leaders, caveats, and verification;
  - `.codex/memory-bank/decisions.md` records the accepted reporting rules for the full-imagery
    comparison: keep protocols separate, treat Stage 5 intervals as pointwise/exploratory unless a
    multiplicity strategy is added, and frame results as weak near-chance effects.
- Final handoff:
  - primary notebook: `notebooks/6.2-full-imagery-model-feature-sweep.ipynb`;
  - primary summary: `artifacts/experiments/full-imagery/stage5_comparison_summary.json`;
  - figures: `artifacts/experiments/full-imagery/stage5_figures/`;
  - use the Stage 5 summary for thesis tables and figures, not the intermediate Stage 3/4 matrix
    summaries alone.
- Verification:
  - `git diff --check` passed;
  - static conflict-marker/placeholder scan over touched memory and Stage 5 files passed.

## Decisions And Assumptions

- Approved by the user on 2026-06-23 after reviewing the draft plan.
- The full sweep targets `Data_Pattern/patt`, not `Data_Train/exec`.
- `geometric` and `random` samples both contribute their 6x6 binary `img` payload to the same
  reconstruction target.
- `sample_type`, `pattern_id`, and `seed` are provenance fields, not model features.
- The classical sweep should report fixed feature-family combinations; using multi-candidate
  feature screening alone would not satisfy the request for all model/feature combinations.
- The current local full `Data_Pattern/patt` index contains 540 rows: 360 geometric and 180 random.
- With the existing cross-subject split seed, the full corpus yields 423 train and 117 test rows
  from 26 and 7 subjects respectively.
- The existing within-subject eligibility yields 27 subjects and 243/243 rows per direction on the
  full corpus.
- Deterministic geometric image repeats are allowed as repeated task labels in the full-imagery
  sweep. Random image fingerprint overlaps, random seed overlaps, sample-key overlaps, and protocol
  boundary violations remain forbidden leakage.

## Progress Log

- 2026-06-23: User approved the staged plan. Saved as approved with Stage 1 pending.
- 2026-06-23: User requested "дальше"; Stage 1 started.
- 2026-06-23: Stage 1 produced the mixed-target contract and full-corpus audit artifact, but the
  plan is blocked because the approved image-payload disjointness contract fails on all 13
  deterministic geometric patterns.
- 2026-06-23: User requested "дальше"; accepted deterministic geometric pattern repeats as labels,
  kept random image/sample-key/random-seed leakage forbidden, regenerated the Stage 1 audit as
  ready, and moved Stage 2 to in progress.
- 2026-06-23: Stage 2 implemented the fixed classical model-feature matrix planner, verified 180
  protocol specs and 270 expected direction runs, passed focused checks, `ruff`, and full `pytest`;
  Stage 2 is awaiting user review.
- 2026-06-23: User requested "дальше"; Stage 2 approved and Stage 3 started.
- 2026-06-23: Stage 3 executed the full classical matrix. 167/180 protocol specs completed,
  249 direction run directories validated, and 13 SVM/ElasticNet convergence-blocked protocol specs
  were logged explicitly; Stage 3 is awaiting user review.
- 2026-06-23: User requested "дальше"; Stage 3 approved and Stage 4 started.
- 2026-06-23: Stage 4 executed the full Torch spectral matrix. 24/24 protocol specs completed,
  36 direction run directories validated, reuse validation passed, and the failure log is empty;
  Stage 4 is awaiting user review.
- 2026-06-24: User requested "дальше"; Stage 4 approved and Stage 5 started.
- 2026-06-24: Stage 5 built and executed the comparison notebook/report. The summary covers
  191 completed learned protocol conditions, validates paired-compatible splits for all completed
  runs, writes Stage 5 JSON/figures, and passes focused checks, `ruff`, full `pytest`, and
  `git diff --check`; Stage 5 is awaiting user review.
- 2026-06-24: User requested "дальше"; Stage 5 approved, Stage 6 final memory handoff completed,
  and the full-imagery model-feature sweep plan was marked completed.
