# Glossary

- EEG: Electroencephalography signal recordings.
- EOG: Electrooculography recordings, often useful for eye-movement or artifact context.
- FIF: MNE/Fiff file format used for electrophysiology data.
- Leakage: Any information path that lets validation/test performance benefit from data that should be unavailable at inference time.
- Subject-wise split: A split where all recordings from a subject are assigned to only one partition.
- Leakage-aware evaluation: thesis-facing Russian term is "оценка с контролем утечек" or
  "пайплайн с контролем утечек". Introduce the English term once in parentheses only when needed.
- Train-only transform: thesis-facing Russian term is "преобразование, настраиваемое только по
  обучающей части" or shorter "настройка только по обучающей части". Use English `train-only`
  only at first definition or in implementation memory.
- Cross-subject protocol: thesis-facing Russian term is "межсубъектный протокол"; introduce as
  "межсубъектный протокол (cross-subject)".
- Identity-overlapping bidirectional cross-trial protocol: thesis-facing Russian term is
  "двунаправленный межпробный протокол с совпадающими испытуемыми"; after first definition use
  "двунаправленный межпробный протокол".
- Subject-cluster bootstrap: thesis-facing Russian term is "субъектно-кластерный bootstrap".
- Paired bootstrap comparison: thesis-facing Russian term is "парное bootstrap-сравнение".
- Seeded Bernoulli baseline: thesis-facing Russian term is "бернуллиевская стратегия с
  фиксированным seed".
- Bit accuracy: thesis-facing Russian term is "битовая точность".
- Exact match accuracy: thesis-facing Russian term is "точность полного совпадения".
- Balanced accuracy: thesis-facing Russian term is "сбалансированная точность"; introduce
  the main metric as "попиксельная сбалансированная точность (per-pixel balanced accuracy)".
- Baseline model: thesis-facing Russian term is "базовая модель"; use "базовая стратегия"
  for non-EEG prediction rules.
- Sample key: thesis-facing Russian term is "ключ наблюдения".
- Random seed: thesis-facing Russian term is "случайное зерно" unless referring to a named
  implementation field.
- Schema-v2/schema-v3: implementation-only artifact vocabulary; do not use in thesis-facing prose.
