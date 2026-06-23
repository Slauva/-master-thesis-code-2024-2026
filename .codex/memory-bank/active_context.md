# Active Context

## Current Focus

BNCI2014_009 P300 benchmark staged plan:

- Approved plan saved at `.codex/memory-bank/plans/2026-06-22-bnci2014-009-p300-benchmark.md`.
- Current stage: Stage 7, Final Notebook, Report, And Memory Update, awaiting review after
  implementation and verification on 2026-06-23.
- Scope: leakage-aware `Target` vs `NonTarget` P300 benchmark over BNCI2014_009 with classical
  ERP/xDAWN/Riemannian baselines first, raw ERP deep-learning models next, and spectral deep models
  treated as exploratory.
- Stage 1 deliverables: executed `notebooks/7.2-bnci2014-009-dataset-audit.ipynb` and
  `artifacts/experiments/bnci2014_009/stage1_dataset_audit.json`.
- Stage 1 verification: notebook executed top-to-bottom with marker
  `BNCI2014_009_STAGE1_AUDIT_VERIFIED`; search found no stored error outputs; `uv run ruff check .`
  and `git diff --check` passed.
- Stage 1 result: local MOABB metadata reports 10 subjects, P300 events `Target=2` and
  `NonTarget=1`, interval `[0, 0.8]`, 16 EEG channels, 256 Hz source sampling, and MOABB P300
  filter band `[1, 24]` Hz. Subject 1 loaded as 1,728 finite `float64` epochs with shape
  `(1728, 16, 206)`, sessions `0`-`2`, run `0`, 288 `Target` epochs, and 1,440 `NonTarget`
  epochs.
- Stage 2 deliverables: `experiments/bnci2014_009/__init__.py`,
  `experiments/bnci2014_009/config.py`, `experiments/bnci2014_009/data.py`,
  `confs/experiments/bnci2014_009.yaml`, and `tests/experiments/test_bnci2014_009_data.py`.
- Stage 2 verification: focused tests passed with 5 tests; `uv run ruff check .` and
  `git diff --check` passed; real two-subject smoke produced `(3456, 16, 206)` `float32` epochs
  and two leak-free LOSO splits; real full-dataset smoke produced `(17280, 16, 206)` `float32`
  epochs, subjects 1-10, ten leave-one-subject-out splits, no forbidden leakage, both classes
  present in every train/test partition, and held-out subject counts of 288 `Target` and 1,440
  `NonTarget` epochs per fold.
- Stage 2 operational note: MOABB downloaded missing BNCI2014_009 `.mat` files into `~/mne_data`;
  no raw data was written to the repository.
- Stage 3 deliverables: `experiments/bnci2014_009/features.py`, updated
  `experiments/bnci2014_009/__init__.py`, `tests/experiments/test_bnci2014_009_features.py`, and
  `artifacts/experiments/bnci2014_009/stage3_adapter_smoke.json`.
- Stage 3 verification: focused BNCI2014_009 tests passed with 9 tests; `uv run ruff check .` and
  `git diff --check` passed; real subject-1 smoke built ERP features of shape `(1728, 896)` and
  xDAWN+tangent features with train side `(1152, 36)` and apply side `(576, 36)`.
- Stage 3 result: ERP features are label-free decimated waveform plus mean-amplitude windows.
  xDAWN/tangent-space support is available through an explicit train/apply helper and must be fit
  only inside training folds in Stage 4.
- Stage 4 deliverables: `experiments/bnci2014_009/metrics.py`,
  `experiments/bnci2014_009/baselines.py`, `experiments/bnci2014_009/workflow.py`, updated
  BNCI2014_009 config/exports, focused classical tests, and immutable run artifact
  `artifacts/experiments/bnci2014_009/classical-sweep/7b7a88206dd8d8a5/`.
- Stage 4 verification: focused BNCI2014_009 tests passed with 12 tests; `uv run ruff check .`,
  full-corpus `execute_classical_benchmark(load_bnci009_config(), reuse_existing=True)`,
  `validate_classical_manifest(...)`, and `git diff --check` passed.
- Stage 4 result: full-corpus 17,280-epoch, 10-fold LOSO classical sweep found best mean balanced
  accuracy for `erp-logreg` at `0.7640624999999999` (std `0.058843103679993326`), followed by
  `erp-ridge` at `0.7605555555555557` and `xdawn-tangent-logreg` at `0.7554861111111111`.
  `erp-logreg` mean target recall was `0.7003472222222222`, mean ROC-AUC `0.8506896219135804`, and
  mean PR-AUC `0.5940665572712607`.
- Stage 4 operational note: `erp-linear-svm` uses deterministic `SGDClassifier(loss="hinge")`
  because two full-corpus attempts with sklearn `LinearSVC`/liblinear were interrupted for
  excessive runtime.
- Stage 5 deliverables: `experiments/bnci2014_009/torch_raw.py`, updated BNCI2014_009
  config/exports, focused raw Torch tests, and immutable run artifact
  `artifacts/experiments/bnci2014_009/raw-erp-torch/7afe116a224e20e2/`.
- Stage 5 verification: focused BNCI2014_009 tests passed with 18 tests; `uv run ruff check .`,
  full-corpus `execute_raw_torch_benchmark(load_bnci009_config(), reuse_existing=True)`,
  `validate_raw_torch_manifest(...)`, full `uv run pytest` with 472 passed and 2 pre-existing
  multiprocessing warnings, and `git diff --check` passed.
- Stage 5 result: full-corpus 17,280-epoch, 10-fold LOSO raw ERP deep sweep found best mean
  balanced accuracy for `deep-convnet-raw-erp` at `0.6161458333333332` (std
  `0.061481330050920734`), followed by `raw-cnn-raw-erp` at `0.5795138888888889`,
  `shallow-convnet-raw-erp` at `0.5582986111111111`, and `eegnet-raw-erp` at
  `0.5494791666666666`. The best raw deep result remains below Stage 4 `erp-logreg` by about
  `0.1479166666666667` balanced-accuracy points.
- Stage 5 operational note: raw ERP tensors use shape `(epoch, 1, channel, time)`, train-fit-only
  standardization, validation subjects drawn only from training subjects, balanced cross-entropy,
  and compact untuned one-seed training. Model checkpoints are not persisted.
- Stage 6 deliverables: `experiments/bnci2014_009/torch_spectral.py`, updated BNCI2014_009
  config/exports, focused spectral Torch tests, and immutable run artifact
  `artifacts/experiments/bnci2014_009/spectral-torch/8d4c3434245e4841/`.
- Stage 6 verification: focused BNCI2014_009 tests passed with 21 tests; `uv run ruff check .`,
  full-corpus `execute_spectral_torch_benchmark(load_bnci009_config(), reuse_existing=True)`,
  `validate_spectral_torch_manifest(...)`, full `uv run pytest` with 475 passed and 2 pre-existing
  multiprocessing warnings, and `git diff --check` passed. `comparison.json` confirms exact
  Stage 5 raw-run test-index alignment.
- Stage 6 result: full-corpus 17,280-epoch, 10-fold LOSO FFT spectral deep sweep found best mean
  balanced accuracy for `eegnet-fft-spectral` at `0.5526736111111111` (std
  `0.0452594676891947`), followed by `deep-convnet-fft-spectral` at `0.5504166666666668` and
  `shallow-convnet-fft-spectral` at `0.5394097222222223`. The best FFT spectral result remains
  below Stage 5 `deep-convnet-raw-erp` by about `0.06347222222222215` and below Stage 4
  `erp-logreg` by about `0.2113888888888888` balanced-accuracy points.
- Stage 6 operational note: FFT log-power tensors use shape `(epoch, 1, channel, frequency)`.
  Morlet, Superlet, and STFT are deferred because the existing time-frequency contracts were
  designed for longer epochs; P300-specific windows should be validated separately.
- Stage 7 deliverables: executed `notebooks/7.3-bnci2014-009-benchmark.ipynb`,
  `artifacts/experiments/bnci2014_009/stage7_benchmark_summary.json`, focused notebook validation
  test, and updated memory bank notes.
- Stage 7 verification: notebook executed top-to-bottom with marker
  `BNCI2014_009_BENCHMARK_VERIFIED`; summary JSON reports best overall `erp-logreg`; focused
  BNCI2014_009 tests plus notebook validation passed with 22 tests; `uv run ruff check .`, full
  `uv run pytest` with 476 passed and 2 pre-existing multiprocessing warnings, and
  `git diff --check` passed.
- Stage 7 result: final notebook confirms `erp-logreg` is the strongest completed benchmark
  (`0.7640624999999999` mean balanced accuracy), followed by the best raw ERP deep model
  `deep-convnet-raw-erp` (`0.6161458333333332`) and the best FFT spectral model
  `eegnet-fft-spectral` (`0.5526736111111111`). Interpret neural results as compact untuned
  exploratory baselines.
- Next action after user approval: mark Stage 7 completed, set the BNCI2014_009 staged plan to
  complete, and provide the final plan summary. Do not proceed without explicit approval.

BNCI2014_009 P300 benchmark Stage 1 historical note:

- Stage 1 was implemented and verified on 2026-06-22, then approved by the user's "дальше" request
  on 2026-06-22.
- Stage 2 was implemented and verified on 2026-06-22, then approved by the user's "дальше" request
  on 2026-06-22.
- Stage 3 was implemented and verified on 2026-06-22, then approved by the user's "дальше" request
  on 2026-06-22.
- Stage 4 was implemented and verified on 2026-06-22, then approved by the user's "Продолжи план"
  request on 2026-06-23.
- Stage 5 was implemented and verified on 2026-06-23, then approved by the user's "дальше" request
  on 2026-06-23.
- Stage 6 was implemented and verified on 2026-06-23, then approved by the user's "следующий шаг"
  request on 2026-06-23.

External EEG dataset feasibility note:

- Current `random_imagery` experiment pipeline is not directly portable as-is because it assumes
  project `patt/random` records, a fixed `[0.5, 15.5)` crop, and 36 binary 6x6 image targets.
- Reusable pieces for global datasets are the MNE/NumPy loading style, leakage-aware subject/trial
  metadata, spectral transforms, classical feature extraction, grouped splits, artifact hashing, and
  reporting discipline.
- A portable benchmark run should first add a task adapter that yields `(eeg, sfreq, channels,
  subject, session/trial, epoch_id, y)` and task-specific crop/metrics, then reuse preprocessing and
  model backends where epoch length permits.

BNCI2014_001 staged plan:

- Approved plan saved at `.codex/memory-bank/plans/2026-06-22-bnci2014-001-pipeline.md`.
- Current stage: Stage 7, Final Notebook, Report, And Memory Update, awaiting review after
  implementation and verification on 2026-06-22.
- Scope: final read-only benchmark notebook/report over the Stage 1, Stage 4, Stage 5, and Stage 6R
  artifacts, with conservative interpretation and no new model training.
- Stage 1 deliverables: executed `notebooks/7.0-bnci2014-001-dataset-audit.ipynb` and
  `artifacts/experiments/bnci2014_001/stage1_dataset_audit.json`.
- Stage 1 verification: subject 1 loaded through MOABB with 576 epochs, 22 EEG channels, 1001 time
  points, balanced four-class labels, sessions `0train`/`1test`, runs `0`-`5`, and raw 250 Hz
  recordings with 22 EEG, 3 EOG, and 1 stim channel. `uv run ruff check .` and `git diff --check`
  passed.
- Stage 2 deliverables: `experiments/bnci2014_001/config.py`,
  `experiments/bnci2014_001/data.py`, `experiments/bnci2014_001/__init__.py`,
  `confs/experiments/bnci2014_001.yaml`, and
  `tests/experiments/test_bnci2014_001_data.py`.
- Stage 2 verification: focused tests passed with 5 tests; `uv run ruff check .` and
  `git diff --check` passed; real two-subject smoke produced `(1152, 22, 1001)` epochs; real
  full-dataset smoke produced `(5184, 22, 1001)` float32 epochs, subjects 1-9, nine
  leave-one-subject-out splits, no forbidden leakage, and all four classes present in every
  train/test partition.
- Stage 3 deliverables: `experiments/bnci2014_001/features.py`,
  `experiments/bnci2014_001/spectral.py`, updated BNCI config exports, focused adapter tests, and
  `artifacts/experiments/bnci2014_001/stage3_adapter_smoke.json`.
- Stage 3 verification: focused BNCI tests passed with 9 tests; `uv run ruff check .` and
  `git diff --check` passed; adapters trim MOABB's 1001-sample epoch to a half-open 1000-sample
  `[0, 4.0)` interval; real subject-1 smoke produced feature matrix shape `(2, 594)`, FFT power
  shape `(22, 39)`, STFT power shape `(22, 39, 8)`, and logical full-corpus payload estimates of
  about 17.74 MiB for FFT and 136.67 MiB for STFT arrays.
- Stage 4 deliverables: `experiments/bnci2014_001/baselines.py`,
  `experiments/bnci2014_001/metrics.py`, `experiments/bnci2014_001/workflow.py`, updated
  BNCI config/exports, focused baseline tests, and immutable run artifact
  `artifacts/experiments/bnci2014_001/csp-lda/447f714497ed180e/`.
- Stage 4 verification: focused BNCI tests passed with 13 tests; `uv run ruff check .`,
  full-dataset `execute_csp_lda_baseline(load_bnci_config(), reuse_existing=True)`,
  `validate_baseline_manifest(...)`, and `git diff --check` passed.
- Stage 4 result: full-corpus 5,184-epoch, 9-fold LOSO CSP+LDA baseline produced mean balanced
  accuracy `0.3852237654320987` (std `0.10863131589647801`) and mean macro F1
  `0.3348272726634433` (std `0.12610162297656752`).
- Stage 5 deliverables: `experiments/bnci2014_001/project_features.py`, updated
  `experiments/bnci2014_001/workflow.py`, updated BNCI config/exports, focused project-feature
  tests, and immutable run artifact
  `artifacts/experiments/bnci2014_001/feature-logreg/902edd18de1c7e5b/`.
- Stage 5 verification: focused BNCI tests passed with 16 tests; `uv run ruff check .`,
  full-dataset `execute_feature_logreg_benchmark(load_bnci_config(), reuse_existing=True)`,
  `validate_baseline_manifest(...)`, and `git diff --check` passed. `comparison.json` confirms
  Stage 5 splits matched Stage 4 CSP+LDA by fold name plus train/test index hashes.
- Stage 5 result: project `time+spectral` features produced matrix shape `(5184, 594)`; full-corpus
  9-fold LOSO one-vs-rest Logistic Regression produced mean balanced accuracy
  `0.3503086419753087` (std `0.08830410888220815`) and mean macro F1 `0.31290918212946645`
  (std `0.095943983880703`), `0.03491512345678999` below the Stage 4 CSP+LDA mean.
- Stage 6 deliverables: `experiments/bnci2014_001/torch_pilot.py`, updated
  `experiments/bnci2014_001/workflow.py`, updated BNCI config/exports, focused Torch pilot tests,
  and immutable run artifact `artifacts/experiments/bnci2014_001/fft-cnn-pilot/b54f16eff24c0908/`.
- Stage 6 verification: focused BNCI tests passed with 19 tests; `uv run ruff check .`,
  full-dataset `execute_torch_fft_pilot(load_bnci_config(), reuse_existing=True)`,
  `validate_baseline_manifest(...)`, and `git diff --check` passed. Tests and workflow preserve the
  boundary that test FFT tensors are materialized only after fold training completes.
- Stage 6 result: full-corpus 9-fold LOSO exploratory FFT-CNN pilot produced mean balanced accuracy
  `0.2633101851851852` (std `0.017836846073405598`) and mean macro F1
  `0.17191417111806218` (std `0.0470451783253082`), below both Stage 4 CSP+LDA and Stage 5
  feature-logreg. Interpret as an intentionally lightweight pre-revision neural smoke/pilot only.
- Stage 6R deliverables: `experiments/bnci2014_001/torch_full.py`, updated BNCI config/exports,
  focused full-Torch tests, tensor caches under
  `artifacts/experiments/bnci2014_001/torch-full-tensors/02d63e3c8ee8372d/`, and immutable run
  artifact `artifacts/experiments/bnci2014_001/torch-full/02d63e3c8ee8372d/`.
- Stage 6R verification: full 12-variant run completed; artifact manifest validation passed;
  focused BNCI test set passed with 24 tests; `uv run ruff check .` and `git diff --check` passed.
- Stage 6R result: best variant was `deep-convnet-stft-bnci` with mean balanced accuracy
  `0.3080632716049383` (std `0.053767218232652424`) and mean macro F1 `0.23715464451660015`.
  Other strong variants were `deep-convnet-morlet-bnci` (`0.3061342592592593`) and
  `deep-convnet-superlet-bnci` (`0.3053626543209877`). The best full Torch variant remains below
  Stage 4 CSP+LDA by `0.07716049382716038` and below Stage 5 feature-logreg by
  `0.042245370370370405`.
- Stage 7 deliverables: executed `notebooks/7.1-bnci2014-001-benchmark.ipynb`,
  `artifacts/experiments/bnci2014_001/stage7_benchmark_summary.json`, and updated plan/memory
  files.
- Stage 7 verification: notebook executed top-to-bottom with marker
  `BNCI2014_001_BENCHMARK_VERIFIED` and no error outputs; strict summary JSON reports best overall
  `csp-lda` and best Torch `deep-convnet-stft-bnci`; focused BNCI tests passed with 24 tests;
  `uv run ruff check .` and `git diff --check` passed.
- Stage 7 result: final notebook confirms CSP+LDA remains the strongest BNCI2014_001 benchmark
  (`0.3852237654320987` mean balanced accuracy), followed by project feature Logistic Regression
  (`0.3503086419753087`) and the best full Torch model `deep-convnet-stft-bnci`
  (`0.3080632716049383`). Interpret the Torch sweep as an untuned exploratory transfer check.
- Next action after user approval: mark Stage 7 completed, set the BNCI2014_001 staged plan to
  complete, and provide the final plan summary. Do not proceed without explicit approval.

Thesis advisor comments staged plan:

- Approved plan saved at
  `.codex/memory-bank/plans/2026-06-22-thesis-advisor-edits.md`.
- Current stage: Stage 7, Static QA And Optional Build, awaiting review after user approved Stage 6
  and moving to the next stage on 2026-06-22.
- Stage 1 deliverables:
  `../latex/chapters/litreview/brain-activity-reconstruction.tex`,
  `../latex/biblio/bibliography.bib`, and `../latex/biblio/footcite-entries.tex`.
- Stage 1 verification: `rg` confirmed the four new arXiv citation keys exist in both
  `bibliography.bib` and `footcite-entries.tex`, and Section 1.4 now names EEG2IMAGE,
  DreamDiffusion, BrainVis, and GWIT while preserving the caveat that generative EEG-to-image
  pipelines do not directly solve the small binary visual-imagery task.
- Stage 2 deliverables:
  `../latex/chapters/introduction.tex`, `../latex/chapters/methodology/base.tex`,
  `../latex/chapters/experiments/base.tex`, and `../latex/chapters/conclusion.tex`.
- Stage 2 verification: `rg` confirmed the old negative-result lead is absent and the new
  strict-evaluation framing appears in the introduction, experiments, and conclusion. No LaTeX build
  was run.
- Stage 3 deliverables:
  `../latex/chapters/methodology/base.tex`, `../latex/chapters/experiments/base.tex`, and
  `.codex/memory-bank/decisions.md`.
- Stage 3 verification: executable dataset indexing from `code/` reported 540 prepared
  recollection-phase blocks with 180 random and 360 geometric blocks; edited thesis-facing
  methodology/experiments fragments contain no internal storage names `Data_Pattern`, `patt`,
  `.fif`, `labels.json`, or `img`; duplicate-label search reported none. After the user reported a
  build error, an explicit `\selectlanguage{russian}` was added before the methodology section
  "Стратегии обучения и валидации"; the approved Docker/latexmk diploma build completed
  successfully and produced `../latex/output/diploma.pdf` with 91 A4 pages. Log search found no
  LaTeX errors, undefined citations/references, fatal errors, or `Command \CYR... unavailable`
  encoding errors.
- Stage 4 deliverables:
  `../latex/images/eeg_rhythms_frequency_bands.png` and
  `../latex/chapters/litreview/basics-eeg-visual.tex`.
- Stage 4 verification: regenerated PNG was visually inspected, is a valid 1500x1080 RGB PNG, and
  contains neutral academic labels; `rg` found no `подсозн` or `доступ к подсознанию` in
  thesis-facing text; Docker/latexmk rebuilt `../latex/output/diploma.pdf` successfully with 91 A4
  pages; log search found no LaTeX errors, undefined citations/references, fatal errors, or
  `Command \CYR... unavailable` encoding errors.
- Stage 5 deliverables:
  `../latex/chapters/introduction.tex`, `../latex/chapters/methodology/base.tex`,
  `../latex/chapters/experiments/base.tex`, `../latex/chapters/conclusion.tex`,
  `../latex/chapters/appendix/base.tex`, selected Chapter 1 literature files, and
  `.codex/memory-bank/glossary.md`.
- Stage 5 verification: `schema-v*` no longer appears in thesis-facing LaTeX files; the remaining
  flagged English terms are deliberate first definitions or non-user-facing `cross-subject` labels;
  duplicate-label search reported none; Docker/latexmk rebuilt `../latex/output/diploma.pdf`
  successfully with 91 A4 pages; log search found no LaTeX errors, undefined citations/references,
  fatal errors, or `Command \CYR... unavailable` encoding errors.
- Stage 6 deliverables:
  new `Связь с предшествующей работой` section and `tab:predecessor-comparison` in
  `../latex/chapters/methodology/base.tex`, plus nearby terminology cleanup in
  `../latex/chapters/methodology/base.tex` and `../latex/chapters/experiments/base.tex`.
- Stage 6 verification: `dementyev2025visualstimuli` is cited in the comparison section and exists
  in both `bibliography.bib` and `footcite-entries.tex`; duplicate-label search reported none;
  targeted terminology search found no visible `Trial`, `upstream`, `schema-v*`, or
  `visual imagery reconstruction` in the edited methodology/experiments files, with only the
  intentional first-definition `cross-trial` term remaining; Docker/latexmk rebuilt
  `../latex/output/diploma.pdf` successfully with 93 A4 pages; log search found no LaTeX errors,
  undefined citations/references, fatal errors, or `Command \CYR... unavailable` encoding errors.
- Stage 7 deliverables:
  final static QA for thesis-facing LaTeX sources, a cleanup of the deep
  `experiments/random_imagery_torch` implementation URL in `../latex/chapters/appendix/base.tex`,
  and an updated `../latex/output/diploma.pdf`.
- Stage 7 verification: static search found no placeholders, `schema-v*`, internal dataset storage
  names, `.fif`, `labels.json`, notebook paths, `experiments/`, `outputs/`, `upstream`,
  `visual imagery reconstruction`, or removed Figure 1.3 wording in thesis-facing LaTeX.
  Structural validator reported 32 labels with no duplicates, no missing references, 17
  `\includegraphics` targets with no missing files, and 43 citation keys with no missing BibTeX or
  full-footcite entries. Docker/latexmk rebuilt `../latex/output/diploma.pdf` successfully with 93
  A4 pages; log search found no LaTeX errors, undefined citations/references, fatal errors, or
  `Command \CYR... unavailable` encoding errors. Build was run despite being optional because the
  final appendix URL cleanup changed a thesis source file.
- Post-Stage-7 annotation update: user requested annotation synchronization with the advisor-edit
  changes. `../latex/annotation.tex` now reflects the inherited 2025 dataset, strict leakage-aware
  evaluation framing, 180-row random-imagery subset, 15-second epoch-level window defense, and
  conservative results. Docker/latexmk rebuilt `../latex/output/annotation.pdf` successfully with 5
  A4 pages; log search found no LaTeX errors, undefined citations/references, fatal errors, or
  `Command \CYR... unavailable` encoding errors; annotation has 3 citation keys with no missing
  BibTeX or full-footcite entries.
- Scope: targeted thesis revisions responding to the scientific advisor's comments on Section 1.4
  literature specificity, leakage-aware framing, 15-second window justification, Figure 1.3,
  terminology consistency, removal of implementation jargon, dataset-count accounting, and explicit
  comparison with the 2025 Dementyev/Parepko/Baranov thesis.
- Constraint: execute one stage at a time and stop at `Awaiting Review`; do not proceed to the next
  stage without explicit approval.

Thesis writing staged plan:

- Approved plan saved at
  `.codex/memory-bank/plans/2026-06-16-thesis-methodology-experiments-appendix.md`.
- Current stage: Stage 8, Static QA And Memory Update, completed.
- Stage 1 deliverable:
  `.codex/memory-bank/thesis-writing-evidence.md`.
- Stage 2 deliverable:
  `../latex/chapters/methodology/base.tex`.
- Stage 3 deliverable:
  `../latex/chapters/appendix/base.tex`.
- Stage 4 deliverable:
  deep-learning architecture appendix in `../latex/chapters/appendix/base.tex`.
- Stage 5 deliverable:
  rewritten `../latex/chapters/experiments/base.tex`.
- Stage 6 deliverable:
  thesis-facing experiment tables in `../latex/chapters/experiments/base.tex`.
- Stage 7 deliverables:
  `scripts/generate_thesis_figures.py` and nine generated PDF figures in `../latex/images/`.
- Stage 8 deliverables:
  final static QA report in the staged plan and updated durable memory.
- Latest progress: the user's 2026-06-16 request to continue the plan was treated as approval of
  Stage 7. Stage 8 completed final static QA for the thesis chapter set, cleaned one false-positive
  LaTeX quote/backtick fragment in the appendix, rechecked placeholders, stale internal wording,
  fixed-width tables, figure targets, labels, citation keys, Ruff status for the figure generator,
  and evidence traceability for major numerical claims.
- Stage 7 checks: figure generation verified the final comparison anchors
  `ridge-regression-independent:0.518382`,
  `deep-convnet-stft-multilabel:0.512011`, and `min_holm_p=0.273000`;
  PNG previews were visually inspected for the cross-subject ranking, reconstruction examples,
  pipeline, and EEGNet architecture figure; all `\includegraphics` targets exist; new figure labels
  are unique; `uv run ruff check scripts/generate_thesis_figures.py` passes; static searches found
  no `TODO`, `\ldots`, `f_s =`, internal dataset storage names, repository artifact paths, or
  notebook paths in the edited thesis chapter set.
- Stage 8 checks: static search found no `TODO`, `\ldots`, `f_s =`, internal dataset storage
  names, repository artifact paths, notebook paths, backtick-formatted code/path fragments, or
  fixed-width `tabular` environments in the edited thesis chapter set. Verified 22 labels with no
  duplicates, 9 figure targets with no missing files, and 15 citation keys with no missing BibTeX
  entries. `uv run ruff check scripts/generate_thesis_figures.py` passes. Major numerical claims
  were checked against `thesis-writing-evidence.md`, `experiments.md`, and the assert-backed
  figure-generation script.
- Thesis citation style note: all literature references in the LaTeX thesis should use
  `\footcite{...}`. The current BibTeX/GOST setup prints full bibliography entries in footnotes
  through `../latex/biblio/footcite-entries.tex`, while `../latex/chapters/bibliography.tex`
  remains the normal `\bibliographystyle{biblio/gost2008n}` + `\bibliography{biblio/bibliography}`
  list. If new `.bib` entries or citation keys are added, rebuild BibTeX and refresh the
  footcite entries file from the new `.bbl`.
- Next action after user approval: optional LaTeX compilation/build if the user gives separate
  permission. Do not run LaTeX compilation without separate user permission.
- Scope guardrail: the thesis methodology/experiments/appendix staged writing plan is complete;
  future work should be treated as a new request or optional build/review step.

Thesis finalization staged plan:

- Approved plan saved at
  `.codex/memory-bank/plans/2026-06-16-thesis-finalization.md`.
- Current stage: Stage 8, Memory Update And Handoff, completed.
- Final deliverables:
  standalone 5-page annotation source `../latex/annotation.tex`, updated
  `../latex/chapters/introduction.tex`,
  `../latex/chapters/conclusion.tex`, light Chapter 1 consistency edits, hidden PDF link boxes in
  `../latex/settings/preamble.tex`, `../latex/output/diploma.pdf`, and
  `../latex/output/annotation.pdf`.
- Latest progress: thesis finalization plan completed end to end. The annotation placeholder was
  replaced and moved out of the main diploma into a standalone annotation document using the same
  `\footcite{...}` plus GOST/BibTeX literature style as the diploma; the introduction was
  reconciled with conservative results and now explicitly names the object and subject of
  research, the conclusion was added and connected, Chapter 1 was lightly aligned, static QA
  passed, and Docker/latexmk builds produced both final PDFs.
- Final checks: static search found no thesis-facing placeholders, stale internal paths, or
  backtick-formatted path fragments; 30 labels have no duplicates; 17 figure targets exist; 39
  citation keys have BibTeX entries; the LaTeX logs have no errors, undefined references, or
  missing citations. Main `diploma.pdf` metadata reports 88 A4 pages and its TOC excludes
  annotation; standalone `annotation.pdf` metadata reports 5 A4 pages and includes four cited
  literature entries. The bibliography contains 39 entries, satisfying the minimum-30 requirement.
  Residual non-blocking overfull/underfull warnings remain, mostly from long English terms,
  bibliography entries, and existing thesis sections.
- Next action: user/scientific-advisor review of the built PDF.

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

PyTorch spectral random-imagery models:

- The approved staged plan is stored in
  `.codex/memory-bank/plans/2026-06-15-torch-spectral-random-imagery-models.md`.
- Stage 1, crop-aware spectral input contracts and train-only normalization, is completed.
- Stage 2, five PyTorch architecture ports, max-norm utilities, attribution, and structural
  verification, is completed.
- Stage 3, leakage-safe grouped model selection and three-seed ensemble training, is completed.
- Stage 4, immutable Torch artifacts, workflows, and CLI, is completed.
- Stage 5, full real-corpus training for all 12 primary variants and both protocols, is
  completed.
- Stage 6, final Torch/classical/probability/baseline comparison, is implemented and awaiting
  explicit final review.
- Added `experiments/random_imagery_torch` with a strict `[0.5, 15.5)` crop configuration, a
  separate atomic cache under `artifacts/preprocessed-imagery/`, immutable spectral and
  normalization schemas, train-key-only per-frequency log-power z-scoring, aligned Torch datasets,
  and strict collation.
- Crop caches are distinct from full-recording spectral caches, validate the EEG source signature,
  preprocessing and crop identity, array metadata, channel order, axes, dtype, scaling, and
  source-time reference before reuse.
- FFT model batches use `(batch, 1, electrode, frequency)`. Morlet, Superlet, and STFT use
  `(batch, frequency, electrode, time)`. No EOG tensor enters the model-input API.
- Canonical random-imagery key `(1, 1, 7)` produced FFT `(63, 39)`, Morlet `(63, 39, 49)`,
  Superlet `(63, 39, 46)`, and STFT `(63, 39, 51)`.
- A real cross-subject smoke check fitted normalization on train key `(1, 1, 7)` only and then
  materialized test key `(9, 1, 7)` with exact `(sample, 36)` target alignment.
- Added 10 focused Stage 1 tests covering crop order, cache hit/invalidation/corruption, source
  signatures, strict sample contracts, normalization provenance, model geometry, delayed test
  access, and target-payload alignment.
- Added spectral adaptations of EEGNet, DeepConvNet, ShallowConvNet, EEGNet-SSVEP, and EEGNet-v1.
  Every model declares an exact input shape and emits 36 raw logits for the shared multi-label
  task.
- The ports preserve source filter counts, EEGNet depthwise/separable blocks, activations, dropout,
  and max-norm intent. TensorFlow SAME padding, spectral-width kernel caps, and adaptive global
  pooling make the short frequency/time axes explicit adaptations rather than numerical ports.
- Added deterministic initialization, per-output-filter max-norm projection, architecture factory
  and primary/exploratory groups, strict geometry validation, and parameter-count snapshots.
- Retained the complete ARL EEGModels CC0 1.0/Apache-2.0 license plus modification and citation
  notice as package data. The supplied `eegnet-tesnorflow.py` was removed after verification.
- Added 69 architecture tests. All five models pass CPU forward/backward on FFT, Morlet, Superlet,
  and STFT geometries plus CUDA smoke, max-norm, depthwise-group, deterministic-init, parameter,
  license, and exact-output checks.
- Real canonical CUDA forward/backward passed for all 12 primary model/preprocessing combinations
  on the RTX 3070 Ti.
- Ruff, lockfile validation, wheel/sdist build, diff check, and the full suite pass with 413 tests;
  two existing Python 3.13 multiprocessing `fork()` deprecation warnings remain.
- The primary experiment covers EEGNet, DeepConvNet, and ShallowConvNet crossed with FFT,
  Morlet, Superlet, and STFT.
- Each model jointly predicts 36 pixels and uses a three-seed ensemble.
- Full real-corpus scope is 36 direction runs: 12 variants across one cross-subject and two
  cross-trial directions.
- Added strict `TorchTrainingConfig` defaults for AdamW, batch size 16, no AMP, three grouped
  validation folds, early stopping, gradient clipping, seeds `42/43/44`, and threshold `0.5`.
- Added grouped fold construction with subject-disjoint train/validation rows and both-class
  validation for all 36 pixels.
- Fold normalization and per-pixel positive weights are fitted only from each fold's training
  rows. Final normalization and weights are fitted only from the complete direction-training rows.
- Added finite-loss, finite-gradient, max-norm-after-step training loops, median fold-best epoch
  selection, CPU checkpoint snapshots, and final three-seed ensemble fitting.
- `fit_torch_ensemble(...)` validates the protocol direction and does not materialize outer-test
  spectral tensors. `predict_torch_ensemble(...)` is the separate post-fit boundary that validates
  disjoint test rows and returns finite float64 mean sigmoid scores plus thresholded int8 labels.
- Added 7 focused Stage 3 tests. Focused Torch tests report 86 passed; full verification reports
  420 passed with the two existing Python 3.13 multiprocessing `fork()` warnings.
- Added `TorchExperimentConfig`, default `confs/experiments/random_imagery_torch.yaml`, primary
  12-model ID parsing, versioned Torch run hashing, and the `random-imagery-torch` console/module
  entry point.
- Added immutable Torch run artifacts under
  `artifacts/experiments/random-imagery-torch/<model-id>/<config-hash>/`.
- Torch artifacts persist config, environment, split/leakage metadata, preprocessing identity,
  normalization arrays, training histories, fold diagnostics, three state-dict checkpoints,
  member scores, ensemble scores, predictions, metrics, baselines, and a SHA-256/byte inventory.
- Safe Torch loading validates metadata and arrays without deserializing weights. Trusted replay
  requires `trusted=True`, validates the manifest and sample-key/spectral identity, and then uses
  `torch.load(..., weights_only=True)`.
- Added `execute_torch_protocol(...)` train-or-reuse orchestration with immutable duplicate
  rejection and complete-run reuse validation.
- Added 8 focused Stage 4 artifact/workflow/CLI tests. Focused Torch tests report 94 passed; full
  verification reports 428 passed with the two existing Python 3.13 multiprocessing warnings.
- Added and executed `notebooks/6.0-torch-spectral-models-training.ipynb`; it calls
  `execute_torch_protocol(...)` for every planned Torch variant/protocol and verifies a second
  immutable-reuse pass.
- Published 36 immutable Torch direction runs under
  `artifacts/experiments/random-imagery-torch/`: 12 cross-subject, 12 trial-1-to-trial-2, and
  12 trial-2-to-trial-1.
- The run environment recorded CUDA on `NVIDIA GeForce RTX 3070 Ti`. The complete crop-spectral
  imagery caches for FFT, Morlet, Superlet, and STFT are populated under
  `artifacts/preprocessed-imagery/Data_Pattern/patt/`.
- Cross-subject direction balanced accuracy ranges from `0.486743` to `0.513443` with mean
  `0.500453`. Within-subject direction balanced accuracy ranges from `0.479567` to `0.524497`
  with mean `0.502307`.
- Combined within-subject descriptive leader is `deep-convnet-stft-multilabel`, balanced accuracy
  `0.512011`, 95% subject-bootstrap interval `[0.500668, 0.520872]`. This is descriptive pending
  the Stage 6 multiplicity-aware comparison.
- Visual inspection of the two Stage 5 notebook figures passed. Focused Torch/notebook tests report
  97 passed; Ruff, lockfile check, diff check, and the full suite pass with 429 tests and the two
  existing Python 3.13 multiprocessing warnings.
- Added and executed `notebooks/6.1-torch-classical-comparison.ipynb`; it compares Logistic
  Regression, nine classical schema-v3 variants, 12 Torch spectral variants, and canonical non-EEG
  baselines from immutable artifacts without loading joblib pipelines or Torch checkpoint weights.
- Stage 6 validates exact ordered test sample keys, targets, and subject IDs for every model
  against Logistic Regression. Cross-subject uses 39 held-out rows from seven subjects; combined
  bidirectional cross-trial uses 162 held-out rows from 27 identities.
- Stage 6 uses the same 2,000 subject-cluster bootstrap draws within each protocol and reports
  Holm-adjusted bootstrap p-values across the 21 non-reference learned models. Minimum
  Holm-adjusted balanced-accuracy p-value is `0.273000`.
- Final descriptive leaders are `ridge-regression-independent` cross-subject with balanced
  accuracy `0.518382` and `deep-convnet-stft-multilabel` combined within-subject with balanced
  accuracy `0.512011`.
- No model is promoted as superior to Logistic Regression under the multiplicity-aware paired
  bootstrap screen. Exact 36-pixel reconstruction remains zero for all learned models in the final
  comparison.
- Visual inspection of all six final-comparison figures passed. Focused Stage 6 tests report
  10 passed; Ruff, lockfile check, diff check, and the full suite pass with 430 tests and the two
  existing Python 3.13 multiprocessing warnings.

## Next Actions

- Obtain explicit final approval for PyTorch spectral-model Stage 6, then mark
  `.codex/memory-bank/plans/2026-06-15-torch-spectral-random-imagery-models.md` completed.
- The original Logistic Regression Stage 5 remains awaiting separate review; the extension's
  reconstruction metrics are already incorporated.
- Obtain explicit final approval for EEG feature-extraction Stage 5 separately.
- Benchmark full-corpus cache warmup only when operational timing is needed.

## Open Questions

- No unresolved questions remain for the completed evaluation/CLI extension.
