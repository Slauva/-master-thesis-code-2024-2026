# Thesis Writing Evidence Inventory

Last updated: 2026-06-16
Plan: `.codex/memory-bank/plans/2026-06-16-thesis-methodology-experiments-appendix.md`

## Evidence Boundaries

- This file supports thesis writing only. It is not a replacement for source code, executed
  notebooks, or experiment artifacts.
- Facts below are source-backed by memory-bank entries, executed notebooks, JSON artifacts, or
  checked thesis bibliography/source-map entries.
- Later writing stages must not turn descriptive ranks into statistical superiority claims.

## Core Task Facts

| Fact | Value | Evidence |
| --- | --- | --- |
| Main experimental subset | `Data_Pattern/patt`, records with `type="random"` | `.codex/memory-bank/decisions.md`; `.codex/memory-bank/active_context.md` |
| Target shape | row-major binary `(sample, 36)` from $6 \times 6$ images | `.codex/memory-bank/experiments.md`; experiment arrays |
| Row count | 180 random-imagery rows | `.codex/memory-bank/active_context.md`; `.codex/memory-bank/experiments.md` |
| Main crop | half-open `[0.5, 15.5)` seconds | `.codex/memory-bank/decisions.md`; `confs/features/default.yaml` |
| Feature analysis rate | 125 Hz | `confs/features/default.yaml` |
| Cross-subject split | 141 train rows, 39 test rows from disjoint subjects | `artifacts/experiments/logistic-regression/4fcdf3c4fa5ef75a/evaluation.json` |
| Cross-trial directions | 81 train rows, 81 test rows per direction | `.codex/memory-bank/active_context.md`; Torch/classical artifacts |
| Combined cross-trial evaluation | 162 held-out rows from 27 identities | `.codex/memory-bank/experiments.md`; `notebooks/6.1-torch-classical-comparison.ipynb` |
| Excluded cross-trial subjects | `14, 24, 27, 28, 29, 32` | `.codex/memory-bank/active_context.md` |
| Bootstrap | 2,000 accepted subject-cluster bootstrap draws per protocol | `notebooks/6.1-torch-classical-comparison.ipynb`; evaluation JSON |

## Dataset And Pipeline Sources

| Topic | Source paths | Notes for thesis |
| --- | --- | --- |
| Dataset semantics | `.codex/memory-bank/active_context.md`, `.codex/memory-bank/decisions.md`, standalone semantic layer | Do not infer a train/test split from `Data_Train` and `Data_Pattern` names. In methodology prose, hide internal names and describe the ready visual-imagery/random-stimulus subset. |
| Predecessor dataset and preprocessing | `notes/2025-dementyev-parepko-baranov-visual-stimuli-reconstruction-thesis.md`; `visual_stimuli_reconstruction_thesis.pdf` | Thesis-facing methodology should state that data were already preprocessed: missing-sample interpolation, block-delay correction, microvolt conversion, average reference, FIR 1-40 Hz, ICA with 20 components, ocular-component removal from blink/saccade masks, and EEG reconstruction. |
| Dataset overview | `notebooks/2.0-dataset-overview.ipynb` | Executed overview and metadata audit; use for corpus description only where needed. |
| Feature extraction | `features/classical.py`, `features/local_patterns.py`, `confs/features/default.yaml` | Full-crop feature groups: time, spectral, spatial, local patterns. |
| Spectral preprocessing | `preprocessors/fft.py`, `morlet.py`, `superlet.py`, `stft.py`; notebooks `2.1`-`2.5` | FFT/Welch, Morlet, Superlet, STFT validated on synthetic and canonical real blocks. Citation keys added: `10.1109/TAU.1967.1161901`, `10.1137/0515056`, `10.1038/s41467-020-20539-9`, `gabor1946theory`. |
| PyTorch spectral inputs | `experiments/random_imagery_torch/spectral_dataset.py`, `torch_dataset.py` | Torch models use crop-specific spectral inputs, not full-recording spectral caches. |
| Model definitions | `experiments/random_imagery_torch/models.py` | EEGNet, DeepConvNet, ShallowConvNet primary; EEGNet-SSVEP and EEGNet-v1 exploratory. |
| Attribution | `experiments/random_imagery_torch/ARL_EEGMODELS_NOTICE.md` | Describe implementations as spectral adaptations of ARL EEGModels architectures. |

## Feature Groups For Appendix

| Group | Implemented content | Evidence | Citation anchors |
| --- | --- | --- | --- |
| Time features | mean, variance, std, RMS, median, MAD, peak-to-peak, skewness, excess kurtosis, normalized line length, zero-crossing rate, Hjorth mobility, Hjorth complexity | `features/classical.py` | General EEG feature context can cite `10.1088/1741-2552/aab2f2`; no extra citation is strictly required for simple statistics. |
| Spectral features | absolute/relative band powers for delta, theta, alpha, beta, low gamma; total power; dominant frequency; spectral centroid; normalized spectral entropy; spectral-transform appendix examples for FFT/Welch, Morlet, Superlet, STFT | `features/classical.py`; `confs/features/default.yaml`; `preprocessors/*.py`; notebooks `2.1`-`2.5` | EEG spectral/preprocessing context: `10.1109/access.2024.3360328`, visual imagery alpha context: `10.1016/j.cub.2020.04.074`; transform keys: `10.1109/TAU.1967.1161901`, `10.1137/0515056`, `10.1038/s41467-020-20539-9`, `gabor1946theory`. |
| Spatial features | OAS covariance, correlation, symmetric matrix logarithm of covariance | `features/classical.py` | Riemannian/covariance context: `10.1080/2326263x.2017.1297192`; source/connectivity caution: `10.1002/hbm.20745`. |
| Local patterns | LNDP, 1D-LGP, 1D-LBP; complete neighborhoods; 256-bin per-channel histograms for `m=8`; probability mode by default | `features/local_patterns.py`; notebooks `4.1`, `4.2` | Jaiswal--Banka: `10.1016/j.bspc.2017.01.005`. |

## Deep Learning Architecture Evidence

| Architecture | Thesis status | Structure evidence | Notes |
| --- | --- | --- | --- |
| EEGNet | Primary Torch architecture in full comparison | `experiments/random_imagery_torch/models.py`; artifacts under `artifacts/experiments/random-imagery-torch/eegnet-*` | Temporal conv, depthwise spatial conv, separable conv, pooling/dropout, 36-logit head. |
| DeepConvNet | Primary Torch architecture in full comparison | `experiments/random_imagery_torch/models.py`; artifacts under `artifacts/experiments/random-imagery-torch/deep-convnet-*` | Temporal+spatial stem, four convolutional blocks with 25/50/100/200 filters, ELU/max-pool/dropout, 36-logit head. |
| ShallowConvNet | Primary Torch architecture in full comparison | `experiments/random_imagery_torch/models.py`; artifacts under `artifacts/experiments/random-imagery-torch/shallow-convnet-*` | Temporal conv, spatial conv, square nonlinearity, average pooling, log activation, 36-logit head. |
| EEGNet-SSVEP | Implemented/tested exploratory port, not part of 12-model full experiment | `experiments/random_imagery_torch/models.py`; `ARL_EEGMODELS_NOTICE.md` | Describe only in appendix as additional port if needed. |
| EEGNet-v1 | Implemented/tested exploratory port, not part of 12-model full experiment | `experiments/random_imagery_torch/models.py`; `ARL_EEGMODELS_NOTICE.md` | Describe only in appendix as additional port if needed. |

## Deep Learning Citation Status

| Needed source | Current local status | Action for later stages |
| --- | --- | --- |
| EEGNet original Lawhern et al. 2018 | Mentioned in `ARL_EEGMODELS_NOTICE.md`; BibTeX key `10.1088/1741-2552/aace8c` added during Stage 2 revision | Use this key for EEGNet architecture attribution. |
| DeepConvNet/ShallowConvNet Schirrmeister et al. 2017 | Mentioned in `ARL_EEGMODELS_NOTICE.md`; BibTeX key `10.1002/hbm.23730` added during Stage 2 revision | Use this key for DeepConvNet and ShallowConvNet architecture attribution. |
| EEGNet-SSVEP Waytowich et al. 2018 | Mentioned in `ARL_EEGMODELS_NOTICE.md`; no confirmed BibTeX key found during Stage 1 | Add verified BibTeX only if the exploratory architecture is described with citation. |
| EEGNet-related existing thesis key | `10.1109/jsen.2023.3270281` exists | Usable as supporting EEGNet/sliding-window context, but not a substitute for original architecture attribution if original claims are made. |

## Experiment Result Anchors

| Result | Value | Evidence |
| --- | --- | --- |
| Logistic Regression cross-subject balanced accuracy | `0.509990918818879` | `artifacts/experiments/logistic-regression/4fcdf3c4fa5ef75a/evaluation.json` |
| Logistic Regression 95% bootstrap CI | `[0.49638366012821206, 0.5210772884032357]` | same artifact |
| Logistic Regression bit accuracy | `0.5142450142450142` | same artifact |
| Logistic Regression micro IoU | `0.33463414634146343` | same artifact |
| Logistic Regression Hamming loss | `0.48575498575498577` | same artifact |
| Logistic Regression exact match | `0.0` | same artifact |
| Descriptive cross-subject leader | `ridge-regression-independent`, balanced accuracy `0.518382` | `notebooks/6.1-torch-classical-comparison.ipynb`; `artifacts/experiments/random-imagery/ridge-regression-independent/c7605762c2e4c898/evaluation.json` |
| Top Torch cross-subject estimate | `shallow-convnet-morlet-multilabel`, balanced accuracy `0.5134425292764515` | `artifacts/experiments/random-imagery-torch/shallow-convnet-morlet-multilabel/678f75c694c69eb2/evaluation.json`; final comparison notebook |
| Descriptive combined within-subject leader | `deep-convnet-stft-multilabel`, balanced accuracy `0.512011` | `notebooks/6.1-torch-classical-comparison.ipynb` |
| One direction of DeepConvNet STFT | Trial 2 -> Trial 1 balanced accuracy `0.5138857282635625` | `artifacts/experiments/random-imagery-torch/deep-convnet-stft-multilabel/dc842ea4e2983fc6/evaluation.json` |
| Minimum Holm-adjusted p-value in final comparison | `0.273000` | `notebooks/6.1-torch-classical-comparison.ipynb` |
| Exact $6 \times 6$ reconstruction accuracy | `0.0` for every learned model in final comparison | `.codex/memory-bank/experiments.md`; final comparison notebook |

## Executed Notebook Markers

| Notebook | Marker or evidence |
| --- | --- |
| `notebooks/5.3-classical-models-comparison.ipynb` | `CLASSICAL_MODELS_COMPARISON_VERIFIED` |
| `notebooks/6.0-torch-spectral-models-training.ipynb` | `SECOND_REUSE_VERIFIED`, `TORCH_STAGE5_TRAINING_COMPLETE` |
| `notebooks/6.1-torch-classical-comparison.ipynb` | `TORCH_CLASSICAL_COMPARISON_VERIFIED`, summary line with leaders and min Holm p-value |

## Stable Writing Decisions

- Treat `Data_Train/exec` and `Data_Pattern/patt` as recording families or experimental phases, not
  as machine-learning train/test partitions.
- In thesis-facing methodology prose, avoid internal dataset/storage names such as `Data_Pattern`,
  `patt`, `.fif`, `labels.json`, and `img`; keep them only in evidence, code, or artifact
  traceability contexts.
- In the main thesis results, keep cross-subject and combined bidirectional cross-trial protocols
  separate.
- Present classical regressor score outputs as clipped continuous scores, not calibrated
  probabilities.
- Present descriptive model rankings as exploratory because the Holm-adjusted paired bootstrap
  screen does not support reliable superiority over Logistic Regression.
- Do not describe BrainBERT, GAN, or diffusion models as completed experiments.

## Open Items For Later Stages

- Add verified BibTeX for Waytowich et al. 2018 only if the exploratory EEGNet-SSVEP architecture
  is described with a direct original-paper citation.
- Decide whether appendix architecture figures will be generated as separate image files or left as
  `nano-banana prompt` comments in LaTeX.
- Add appendix implementation anchors and short examples for FFT/Welch, Morlet, Superlet, STFT,
  LNDP, 1D-LGP, 1D-LBP, EEGNet, DeepConvNet, and ShallowConvNet.
- For reconstruction examples, sample only from real `test_targets.npy` and `predictions.npy` after
  the user approves the figure/table stage.
