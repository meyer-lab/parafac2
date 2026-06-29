# Replacing SVD with Gram matrix + column normalization in `project_data`

## Background

`project_data()` in `parafac2/utils.py` is called multiple times per PARAFAC2 iteration. Its inner loop computes a projection matrix (polar factor) for each condition. Profiling identified `cp.linalg.svd` as the dominant cost.

## Old approach: thin SVD

For each condition `i`, the polar factor was computed as:

```python
lhs = (A[i] * C) @ B.T           # (n_genes, rank)
M   = mat @ lhs - means @ lhs    # (n_cells, rank) — center by global mean
U, _, Vh = cp.linalg.svd(M, full_matrices=False)
proj = U @ Vh
```

The SVD of an `(n_cells, rank)` matrix through cuSOLVER has high per-call overhead and limited parallelism for tall-thin shapes.

## New approach: Gram matrix + column normalization

```python
mean_C = means @ C                # (1, rank) — precomputed once outside the loop

# per condition:
lhs  = (A[i] * C) @ B.T
M    = mat @ lhs - cp.array((A[i] * mean_C) @ B.T, dtype=cp.float32)
M_f64 = M.astype(cp.float64)
G    = M_f64.T @ M_f64            # (rank, rank) float64 Gram matrix
_, V = cp.linalg.eigh(G)          # right singular vectors
MV   = M_f64 @ V                  # columns ≈ U_svd · S · D  (D = diagonal ±1)
col_norms  = cp.linalg.norm(MV, axis=0, keepdims=True)
safe_norms = cp.where(col_norms > 1e-10, col_norms, 1.0)
proj = ((MV / safe_norms) @ V.T).astype(cp.float32)
```

The expensive `svd(n_cells, rank)` is replaced by:

1. `eigh` on a `(rank, rank)` matrix — negligible cost
2. One `(n_cells, rank) × (rank, rank)` GEMM (`M_f64 @ V`)
3. Column-norm division and a second GEMM (`@ V.T`)

All of these are BLAS-3 operations with high GPU utilisation.

### Why column norms, not `sqrt(eigenvalues)`

`eigh` returns eigenvectors with arbitrary column signs: `V_eigh = V_svd @ D` for some diagonal ±1 matrix `D`. Therefore `MV = M @ V_eigh = U_svd @ S @ D`. Dividing each column by its norm (which equals `S_j`, always positive) gives `U_svd @ D`. Multiplying by `V.T = D @ V_svd.T` then cancels `D`:

```
(U_svd @ D) @ (D @ V_svd.T) = U_svd @ V_svd.T  ✓
```

The polar factor is recovered regardless of `eigh`'s sign convention. Dividing instead by `sqrt(eigenvalues(G))` is problematic: for ill-conditioned `M` the eigenvalues of `G = M.T @ M` can be inaccurate (the condition number is squared), producing wrong singular value estimates and a non-orthonormal `proj`. This was caught by a test that checks `P.T @ P ≈ I` with un-normalized factors.

### Why not QR

`QR(MV)` also yields orthonormal columns, but LAPACK's Householder convention does not guarantee a positive diagonal in `R`. The sign of each column of `Q` is implementation-defined, so `Q @ V.T` may differ from `U @ Vh` by column sign flips — producing a different orthogonal factor and causing the algorithm to diverge.

### Why float64 for the Gram matrix

Forming `G = M.T @ M` in float32 squares the condition number. With `M` entries of order 200 (typical for `mat @ lhs` at this scale), `G` entries reach ~4×10⁷, losing several significant digits in float32. This produced ~12% relative error in the MTTKRPs. Promoting `M` to float64 before computing `G` is cheap — `G` is only `(rank, rank)` = 50×50 — and reduces MTTKRP errors to <4×10⁻⁷ relative.

## Mean centering simplification

The mean term factors as:

```
means @ lhs  =  means @ (A[i] * C) @ B.T
             =  (A[i] * (means @ C)) @ B.T
```

Precomputing `mean_C = means @ C` once (shape `(1, rank)`) outside the loop replaces an O(n_genes × rank) multiply per condition with an O(rank²) one. Computing G without centering M first is tempting (the mean appears only as a rank-1 correction to `L.T @ L`), but risks catastrophic cancellation when per-condition means are close to the global mean — common in normalized scRNA data.

## Performance

Benchmark: 50 conditions × 1000 cells × 1000 genes, rank=50, 30 timing runs (NVIDIA GPU).

| Approach | Mean (ms) | Std (ms) | Speedup |
|---|---|---|---|
| SVD baseline | ~188 | ~4.5 | 1.0× |
| eigh + col-norm | ~72 | ~1.2 | ~2.6× |

MTTKRP relative errors vs. SVD: all < 4×10⁻⁷.

Per-operation breakdown of the new approach (profiled with per-op GPU sync; totals inflate due to sync overhead but relative shares are indicative):

| Operation | Time (ms) | Share |
|---|---|---|
| `lhs` computation | 5.7 | 5% |
| `mat @ lhs` (GEMM) | 2.1 | 2% |
| cast M → float64, form G | 3.1 | 3% |
| `eigh(G)` — rank×rank | **56.4** | **53%** |
| `M_f64 @ V` + col-norms | 5.2 | 5% |
| polar factor `@ V.T` | 3.1 | 3% |
| MTTKRP accumulation | 30.9 | 29% |

`eigh` on the 50×50 Gram matrix is now the single largest cost in the projection step, not the large matmul. The (1000×1000)×(1000×50) GEMM runs in ~1–2 μs on the GPU and contributes negligibly. The next optimization target inside `project_data` is the MTTKRP accumulation (29%), which also contains several GEMMs over `(n_cells, rank)` and `(n_genes, rank)` matrices.
