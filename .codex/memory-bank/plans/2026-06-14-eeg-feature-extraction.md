# EEG Feature Extraction Plan

Status: in_progress
Last updated: 2026-06-14
Next stage: 3 - Jaiswal-Banka Local Patterns (Awaiting Review)

## Goal

Implement reproducible modular EEG feature extraction for the visual-imagery dataset, including
classical time/spectral/spatial features and the LNDP/1D-LGP methods from Jaiswal and Banka (2017).

## Scope

- Extract per-channel spectral, amplitude/time, covariance, correlation, log-covariance, LNDP,
  1D-LGP, and comparison 1D-LBP features.
- Crop the canonical imagery interval before resampling and optionally divide it into configurable
  complete windows.
- Preserve subject, trial, block, channel, and window metadata and expose deterministic sklearn
  matrices without fitting scalers, PCA, selectors, or prediction models.
- Cache derived feature groups atomically under `artifacts/features/`.
- Validate scientific behavior with synthetic signals and an executed real-data notebook.

## Acceptance Criteria

- Default configuration uses 125 Hz analysis, crop `[0.5, 15.5)`, `float32`, no repeated filtering,
  rereferencing, or dataset-wide normalization.
- Feature arrays are finite, reproducible, named, and stable in shape for a fixed configuration.
- LNDP and 1D-LGP reproduce the paper examples with codes 7 and 224.
- Window metadata is sufficient to keep every window from one source block in the same ML fold.
- Ruff and the complete pytest suite pass, and the validation notebook executes without errors.

## Stages

### 1. Contracts And Configuration - Completed

- Objective: Fix configuration, crop/window semantics, result shapes, flatten order, and cache identity.
- Deliverables: typed feature config, YAML defaults, crop/window helpers, modular schemas, focused tests.
- Constraints: crop before resampling; use only complete windows; no feature computation in this stage.
- Verification: focused feature config/schema tests, canonical 16 s and 26 s crop checks, Ruff.
- Completion criteria: invalid configurations and array shapes are rejected; defaults resolve
  deterministically; the default crop produces exactly 15 seconds for both recording families.
- Review gate: Stop and wait for explicit user approval.

Implemented:

- Added frozen OmegaConf/Pydantic configuration with canonical frequency bands, feature-group
  selection, histogram mode, local-pattern neighborhood size, and disabled repeated preprocessing.
- Added exact half-open crop and complete-window layouts. Crop/window durations must resolve to
  integer samples; configured windows may leave an unused tail but are never padded.
- Added modular `FeatureBlock` and `FeatureSet` schemas with finite-value, axis, channel, name, and
  window validation.
- Fixed sklearn flatten order as requested block, then channel, then feature/code. Symmetric channel
  matrices use the upper triangle with `sqrt(2)` scaling on off-diagonal entries.
- Added versioned deterministic feature-config hashing and reserved `artifacts/features/` as a
  generated location.

Verification:

- `uv run pytest tests/features -q`: 27 passed.
- Real canonical samples: both `(63, 16001)` `exec` and `(63, 26001)` `patt` crop to
  `(63, 15000)` using source slice `[500:15500]`.
- Configured 4 s windows with 2 s stride at 125 Hz produce six complete windows and omit the final
  incomplete second.
- `uv run ruff check .`: passed.
- `uv run pytest`: 177 passed, 2 skipped; two existing Python 3.13 multiprocessing `fork()`
  deprecation warnings remain.

### 2. Classical Features - Completed

- Objective: Implement time, band-power, spectral-summary, covariance, correlation, and log-covariance features.
- Deliverables: pure feature functions and synthetic numerical tests.
- Constraints: reuse the established resampling and FFT contracts; use OAS shrinkage and stable SPD eigendecomposition.
- Verification: amplitude, frequency, entropy, correlation, covariance, and finite-output tests.
- Completion criteria: all classical feature families satisfy the stage 1 schemas and synthetic expectations.
- Review gate: Stop and wait for explicit user approval.

Implemented:

- Added a shared crop-before-resample pipeline that produces validated float64 working windows and
  preserves configured absolute time bounds.
- Added 13 per-channel time features: mean, population variance/std, RMS, median, MAD,
  peak-to-peak, population skewness/excess kurtosis, mean absolute first difference, zero-crossing
  rate, and Hjorth mobility/complexity.
- Added absolute and relative power for all configured bands plus total power, dominant frequency,
  spectral centroid, and normalized spectral entropy. Power is integrated by frequency-cell
  overlap from the established one-sided Hann FFT density implementation.
- Added OAS shrinkage covariance, covariance-normalized correlation, and symmetric log-covariance
  from a stable eigenvalue decomposition.
- Defined finite zero sentinels for mathematically undefined constant/zero-signal ratios.
- Added a classical feature orchestrator that respects selected feature groups and stage 1 windowing.

Verification:

- `uv run pytest tests/features -q`: 36 passed.
- Synthetic 10 Hz amplitude-2 and 20 Hz amplitude-1 tones recover expected alpha/beta power,
  total mean-square power, dominant frequency, and centroid.
- Spatial tests verify symmetry, positive OAS covariance, correlation diagonal, and reconstruction
  of covariance from the exponential of log-covariance.
- Canonical `Data_Pattern/patt` key `(1, 1, 1)` produced finite `float32` blocks with shapes
  `(1, 63, 13)`, `(1, 63, 14)`, and three `(1, 63, 63)` matrices.
- `uv run ruff check .`: passed.
- `uv run pytest`: 186 passed, 2 skipped; two existing Python 3.13 multiprocessing `fork()`
  deprecation warnings remain.
- Added and executed `notebooks/4.0-classical-features.ipynb` with synthetic spectral/spatial
  validation and canonical `Data_Pattern/patt` full-epoch versus configurable-window demonstrations.
- Notebook integration checks require stored execution counts, no error outputs, explicit imagery
  source/window configuration, and the `CLASSICAL_FEATURES_VERIFIED` marker.

### 3. Jaiswal-Banka Local Patterns - Awaiting Review

- Objective: Implement LNDP, 1D-LGP, and comparison 1D-LBP per channel.
- Deliverables: vectorized code generation, count/probability histograms, paper-example tests.
- Constraints: `m=8` default, full neighborhoods only, no padding, documented bit order.
- Verification: reproduce LNDP 7 and 1D-LGP 224; histogram mass and boundary tests.
- Completion criteria: exact formulas and histogram contracts are protected by tests.
- Review gate: Stop and wait for explicit user approval.

Implemented:

- Added vectorized per-channel LNDP, 1D-LGP, and comparison 1D-LBP code generation using complete
  chronological neighborhoods and no padding.
- Matched the paper's Figures 3 and 4 ordering: the local window is `P_m, ..., P_0`, the center is
  `P_(m/2)`, and bit zero is the rightmost `P_0` relationship.
- Added configurable raw-count and L1-probability histograms with stable zero-padded code names.
  The default `m=8` produces 256 features per channel.
- Added modular `FeatureBlock` output with shape `(window, channel, code)` and configured output
  dtype.
- Added and executed `notebooks/4.1-local-patterns.ipynb` with paper examples, synthetic
  offset-invariance checks, and full-epoch versus six-window canonical imagery demonstrations.

Verification:

- Published examples reproduce LNDP code `7` and 1D-LGP code `224`.
- Focused local-pattern tests verify valid-center counts, leading-axis preservation, code ranges,
  histogram mass, count/probability modes, and `(1/6, 63, 256)` pipeline shapes.
- Canonical `Data_Pattern/patt` key `(1, 1, 1)` produced finite probability histograms for all
  63 channels in one 15 s epoch and six complete 5 s windows with 2 s stride.
- Both feature notebooks execute top-to-bottom, retain execution counts, contain no error outputs,
  and emit their validation markers.
- `uv run ruff check .`: passed.
- `uv run pytest`: 202 passed, 2 skipped; two existing Python 3.13 multiprocessing `fork()`
  deprecation warnings remain.

### 4. Dataset Cache And Sklearn Export - Pending

- Objective: Integrate feature extraction with `NumpyDataset`, atomic modular caches, and deterministic matrix export.
- Deliverables: `FeatureDataset`, cache manifests, `build_feature_matrix`.
- Constraints: do not mix recording families; preserve source metadata and block grouping.
- Verification: cache reuse/invalidation/corruption tests and integration tests for both families.
- Completion criteria: selected groups round-trip through cache and export with stable names and metadata.
- Review gate: Stop and wait for explicit user approval.

### 5. Scientific Validation - Pending

- Objective: Validate synthetic behavior and inspect one real imagery sample.
- Deliverables: executed notebook with signal, spectral, covariance, and local-pattern visualizations.
- Constraints: no ML performance claims; full-epoch and one configurable-window comparison only.
- Verification: execute notebook top-to-bottom, run Ruff and full pytest suite.
- Completion criteria: notebook checks pass, results are recorded, and repository verification is green.
- Review gate: Stop and wait for explicit user approval.

## Decisions And Assumptions

- `Data_Pattern/patt` is the primary imagery corpus; `Data_Train/exec` is supported only for explicit
  auxiliary comparisons.
- The canonical imagery crop is `[0.5, 15.5)` seconds and is applied before resampling.
- `window_seconds=None` produces one full-crop window. Otherwise both window length and stride are
  explicit and only complete windows are retained.
- Feature groups remain modular and are concatenated only through an explicit export operation.
- Local-pattern histograms are independent per channel; article classification accuracy is not a
  reproduction target.
- All windows from one source block share the same split group; train-fitted transforms remain
  outside feature extraction.

## Progress Log

- 2026-06-14: User explicitly approved implementation of the proposed plan.
- 2026-06-14: Stage 1 started.
- 2026-06-14: Stage 1 implemented and verified; awaiting explicit user approval.
- 2026-06-14: User approved stage 1; stage 2 started.
- 2026-06-14: Stage 2 implemented and verified; awaiting explicit user approval.
- 2026-06-14: User requested and approved an executed demonstration notebook before continuing.
- 2026-06-14: Classical feature notebook executed and inspected; stage 2 completed and stage 3 started.
- 2026-06-14: Stage 3 implemented, executed, visually inspected, and verified; awaiting explicit
  user approval before stage 4.
