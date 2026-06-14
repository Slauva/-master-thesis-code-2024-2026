# Decisions

## 2026-06-10

- Store project operating instructions in `Agent.md`.
- Store persistent project context in `memory-bank/`.
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
