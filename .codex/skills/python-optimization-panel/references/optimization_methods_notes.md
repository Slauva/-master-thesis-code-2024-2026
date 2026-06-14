# Optimization Methods Notes

Source preserved as `optimization_methods.pdf`.

Title: "Методы оптимизации вычислений в Python"
Subtitle: "На примере анализа межсубъектных корреляций (ISC) в ЭЭГ данных"
Created: 2026-02-01

## Scope

Use these notes when optimizing NumPy-heavy scientific Python, especially EEG/ISC code with:

- many subjects, channels, frequency bands, conditions, or time points;
- pairwise correlations;
- leave-one-out computations;
- bootstrap confidence intervals;
- permutation tests;
- repeated channel or metadata lookups.

## Core Principle

Do not write Python loops for work NumPy can do in compiled vectorized operations.
Python should orchestrate logic; NumPy should do numeric computation.
Profile first, then optimize the measured bottleneck.

## Methods

### 1. Vectorization and Broadcasting

Replace nested Python loops over pairs with array masks and index arrays.

- Use `groups[:, None] == groups[None, :]` to build same-group matrices.
- Use `np.triu_indices(n, k=1)` to work only with unique pairs.
- Use boolean masks to split within-group and between-group values.
- Expected speedup in the handout: roughly 50-100x for pairwise ISC-style grouping.

### 2. Precomputation for Leave-One-Out

Avoid recomputing means for every subject.

- Stack data as `(n_subjects, n_channels, n_features)`.
- Compute `total_sum = stacked.sum(axis=0)` once.
- For subject `i`, use `(total_sum - stacked[i]) / (n - 1)`.
- Vectorize per-channel correlations instead of calling `pearsonr` inside a channel loop.
- Complexity improves from approximately `O(n^2 * C * T)` to `O(n * C * T)`.

### 3. Batch Bootstrap

Generate all bootstrap samples in one array instead of looping over bootstrap iterations.

- Use a local RNG with fixed seed for reproducibility.
- Generate indices with shape `(n_bootstrap, n)`.
- Use advanced indexing to create all samples and reduce along axis 1.
- Watch memory: the sample matrix costs `O(n_bootstrap * n)`.

### 4. Dictionaries for Repeated Lookup

Replace repeated `list.index()` or `x in list` checks with a dictionary.

- Build `channel_to_idx = {ch: i for i, ch in enumerate(channel_list)}` once.
- Lookup becomes average `O(1)` instead of `O(n)`.
- Useful for repeated EEG channel group selection.

### 5. Matrix Multiplication for Correlation Matrices

Replace pairwise Pearson calls with normalization plus one matrix multiply.

- Flatten each subject/condition to rows of shape `(n_items, n_features)`.
- Center each row.
- Normalize by row norm with `np.errstate` and `np.nan_to_num`.
- Compute `R = normalized @ normalized.T`.
- Fill diagonal explicitly if needed.
- Handout speedup: roughly 20-200x depending on baseline and BLAS.

### 6. `np.einsum` for Indexed Tensor Algebra

Use `einsum` when nested loops express tensor contractions.

- For within-subject covariance in CCA/ISC: `np.einsum("ijk,ilk->jl", stacked, stacked)`.
- For between-subject covariance, use the identity:
  `sum_{i != j} Xi @ Xj.T = (sum_i Xi) @ (sum_i Xi).T - sum_i Xi @ Xi.T`.
- This can reduce `O(n^2 * C^2 * T)` work to `O(n * C^2 * T)`.

### 7. Optimized Permutation Tests

Precompute everything invariant across permutations.

- Compute `triu_i`, `triu_j`, and `R_upper` once.
- In each permutation, only permute groups and recompute masks.
- Use the corrected p-value formula:
  `(count(abs(null) >= abs(observed)) + 1) / (n_permutations + 1)`.
- For extra speed, generate all permutations up front and vectorize masks, but check memory:
  full vectorization costs `O(n_permutations * n_pairs)`.

## Practical Checklist

- Is there a Python `for` over array elements, channels, pairs, bootstrap samples, or permutations?
  Try vectorization, batching, or precomputing invariant arrays.
- Are the same sums, indices, masks, or channel positions recomputed?
  Cache or precompute them.
- Is code doing many `pearsonr` calls?
  Consider centered/normalized matrix multiplication.
- Is memory growth large after vectorization?
  Use chunking or keep a small loop around large vectorized blocks.
- Are random methods used?
  Use an explicit local RNG and record the seed.
- Did the optimized code preserve statistical semantics?
  Check splits, leakage boundaries, degrees of freedom, and p-value formulas.

## Verification

- Compare optimized output against the simple implementation on small synthetic arrays.
- Include edge cases: constant signals, NaNs/Infs, empty masks, one-subject groups, and imbalanced groups.
- Benchmark with `timeit` or a profiler after correctness tests pass.
- For large EEG arrays, report both runtime and peak memory.
