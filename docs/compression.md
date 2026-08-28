# Compression Tutorial (CANDELINC)

This tutorial explains how to accelerate PARAFAC2 factorizations using **CANDELINC** (CANDECOMP with Linear Constraints) compression. Single-cell datasets often contain hundreds of thousands to millions of cells across dozens of experimental conditions and tens of thousands of genes. Fitting standard PARAFAC2 Alternating Least Squares (ALS) on such datasets requires repeated passes over the entire single-cell data matrix at every iteration.

CANDELINC compression dramatically accelerates fitting by projecting the gene and cell modes into compact orthonormal subspaces _before_ running ALS. The iterative decomposition then runs entirely on small, dense core matrices. Once converged, the full factor matrices and per-cell projections are reconstructed exactly into the original data space without information loss beyond the initial subspace truncation.

---

## How Compression Works

PARAFAC2 decomposes multi-condition single-cell data by modeling slice $k$ ($N_k$ cells by $J$ genes) as:

$$X_k \approx P_k B \operatorname{diag}(A_{k,:}) C^T$$

where $P_k \in \mathbb{R}^{N_k \times R}$ has orthonormal columns ($P_k^T P_k = I_R$), $B \in \mathbb{R}^{R \times R}$, $A \in \mathbb{R}^{K \times R}$, and $C \in \mathbb{R}^{J \times R}$.

CANDELINC compression operates in two steps:

1. **Gene Compression ($L_g$)**: Computes an orthonormal gene basis $Q \in \mathbb{R}^{J \times L_g}$ via randomized SVD on the mean-centered data, yielding gene-compressed matrix $X_c = (X - \mathbf{1}\boldsymbol{\mu}^T) Q$.
2. **Cell Compression ($L_c$)**: Computes per-condition cell bases $Q_k \in \mathbb{R}^{N_k \times L_{c,k}}$ via thin SVD on each condition slice $X_{c,k}$, yielding small dense core matrices $Y_k \in \mathbb{R}^{L_{c,k} \times L_g}$.

During ALS fitting, factor updates and Matricized Tensor Times Khatri-Rao Products (MTTKRP) operate exclusively on $Y_k$. After convergence:

- Gene factors are reconstructed: $C = Q C_L \in \mathbb{R}^{J \times R}$.
- Cell projections are reconstructed: $P_k = Q_k \tilde{P}_k \in \mathbb{R}^{N_k \times R}$.

```
Original Data Slices X_k (N_k × J)
       │
       ▼  Gene compression (randomized SVD -> Q)
Gene-compressed X_c,k (N_k × L_g)
       │
       ▼  Cell compression (per-condition SVD -> Q_k)
Dense Cores Y_k (L_c,k × L_g)  ──►  Fast ALS Iterations on Y_k
                                          │
Reconstructed Factors & Projections  ◄────┘
  C = Q @ C_L
  P_k = Q_k @ P_tilde_k
```

---

## Prerequisites and Dataset Preparation

Before compression, your single-cell dataset should be stored in an [`anndata.AnnData`](https://anndata.readthedocs.io/) object with:

1. Integer condition labels in `adata.obs["condition_unique_idxs"]`.
2. Per-gene mean expression values in `adata.var["means"]` (used for implicit mean-centering without densifying sparse matrices).

You can prepare raw single-cell count matrices using [`parafac2.normalize.prepare_dataset`](api.md#parafac2.normalize.prepare_dataset):

```python
import anndata
from parafac2.normalize import prepare_dataset

# Preprocess raw counts: filter low-count cells/genes, normalize, log-transform,
# and compute condition_unique_idxs and var["means"]
adata = prepare_dataset(raw_adata, condition_name="sample_id", geneThreshold=0.01)
```

For the examples below, let us set up a synthetic multi-condition dataset:

```python
import numpy as np
import anndata
from scipy.sparse import csr_array
from parafac2 import compress_dataset, parafac2_nd, store_pf2

# Generate synthetic 5-condition single-cell data
rng = np.random.default_rng(42)
n_cells, n_genes, n_conds = 3000, 500, 5

# Create sparse count matrix
density = 0.1
raw_mat = (rng.random((n_cells, n_genes)) < density).astype(np.float32)
raw_mat *= rng.exponential(2.0, size=(n_cells, n_genes)).astype(np.float32)
X_sparse = csr_array(raw_mat)

# Assign condition indices and calculate gene means
obs = {"condition_unique_idxs": np.repeat(np.arange(n_conds), n_cells // n_conds)}
var = {"means": np.ravel(X_sparse.mean(axis=0))}
adata = anndata.AnnData(X=X_sparse, obs=obs, var=var)
```

---

## Approach 1: One-Step Inline Compression

The simplest way to use compression is passing the `compress` argument directly to [`parafac2_nd`](api.md#parafac2.parafac2.parafac2_nd). This performs compression and factorization in a single function call.

### Automatic Compression (`compress="auto"` or `compress=True`)

When `compress="auto"` (or `compress=True`), `parafac2_nd` automatically sets the subspace dimensions based on the target `rank`:

$$L_g = \min(n_{\text{genes}}, \max(4 \cdot \text{rank}, \text{rank} + 20))$$
$$L_c = \max(4 \cdot \text{rank}, \text{rank} + 20)$$

```python
# One-step factorization with automatic compression
pf2_output, r2x = parafac2_nd(
    adata,
    rank=15,
    compress="auto",
    random_state=42,
)

# Store weights, factors, and cell projections back into adata
adata = store_pf2(adata, pf2_output)

print(f"Explained variance (R2X): {r2x:.4f}")
print("Projections matrix shape:", adata.obsm["projections"].shape)
```

### Custom Compression Dimensions

You can also specify explicit compression dimensions:

- **Scalar integer (`compress=L`)**: Sets both $L_g = L$ and $L_c = L$.
- **Tuple `(L_g, L_c)`**: Specifies gene and cell subspace dimensions independently.

```python
# Custom symmetric compression dimension
pf2_output, r2x = parafac2_nd(adata, rank=10, compress=40, random_state=42)

# Custom asymmetric dimensions: 80 gene dimensions, 30 cell dimensions per condition
pf2_output, r2x = parafac2_nd(adata, rank=10, compress=(80, 30), random_state=42)
```

> [!TIP]
> A good rule of thumb is setting $L_g$ and $L_c$ to at least $3\times$ to $4\times$ the target factorization rank $R$. Setting $L$ too close to $R$ can constrain the subspace search during ALS, while overly large $L$ provides diminishing accuracy returns.

---

## Approach 2: Pre-Compression for Fast Rank Sweeps

In exploratory analyses, you typically fit models across a range of ranks (e.g., $R \in [5, 10, 15, 20, 30]$) to select an optimal rank via an $R^2X$ curve.

If you pass the raw `AnnData` to `parafac2_nd` inside a loop, the raw dataset is re-compressed on every iteration. Instead, use [`compress_dataset`](api.md#parafac2.compress.compress_dataset) to compress once, and then fit multiple ranks in seconds on the resulting [`CompressedData`](api.md#parafac2.compress.CompressedData) object.

```python
from parafac2 import compress_dataset, parafac2_nd, store_pf2

# 1. Compress once for a maximum expected rank of 30
compressed = compress_dataset(adata, L="auto", rank=30, random_state=42)

# Inspect compression metadata
print(f"Gene dimension (L_g): {compressed.L_g}")
print(f"Max cell dimension (L_c): {compressed.max_cell_dim}")
print(f"Discarded variance fraction: {compressed.lost_var / compressed.norm_tensor:.4%}")

# 2. Rapidly sweep across multiple ranks
rank_results = {}
for r in [5, 10, 15, 20, 25]:
    output, r2x = parafac2_nd(compressed, rank=r, random_state=42)
    rank_results[r] = (output, r2x)
    print(f"Rank {r:2d} -> R2X: {r2x:.4f}")

# 3. Store the chosen rank results into the original AnnData
best_rank = 15
best_output, best_r2x = rank_results[best_rank]
adata = store_pf2(compressed, best_output)
```

> [!NOTE]
> Passing `compressed` directly to [`store_pf2`](api.md#parafac2.parafac2.store_pf2) automatically identifies the referenced `AnnData` object, reconstructs the full-dimensional per-cell projections $P_k = Q_k \tilde{P}_k$, and writes `Pf2_weights`, `Pf2_A`, `Pf2_B`, `Pf2_C`, `projections`, and `weighted_projections` into `adata`.

---

## Approach 3: Gene-Only Compression (`L_c = None`)

In certain datasets, you may want to compress the gene mode while leaving the cell mode uncompressed:

- Conditions have relatively small numbers of cells ($N_k \le L_c$).
- You wish to avoid the overhead of per-condition SVDs during the compression phase.
- You want exact cell representations throughout all ALS steps.

To perform gene-only compression, pass `(L_g, None)` for the compression parameter:

```python
# Inline gene-only compression
pf2_output, r2x = parafac2_nd(
    adata,
    rank=10,
    compress=(100, None),
    random_state=42,
)

# Or pre-compress with gene-only compression
compressed_gene_only = compress_dataset(
    adata,
    L=(100, None),
    random_state=42,
)
```

When $L_c$ is `None`, each core matrix $Y_k$ has shape $(N_k, L_g)$ (i.e. the uncompressed cell count for condition $k$ by $L_g$ gene components).

---

## Approach 4: Slice Normalization with Compression

Single-cell studies often exhibit imbalance where certain conditions or patient samples contain significantly more cells or higher total variance than others. Without normalization, large conditions can dominate shared factor estimation (such as the condition factor matrix $A$).

The `normalize_slices=True` option weights each condition's ALS update by the inverse of its Frobenius norm ($w_k = 1 / \|X_k\|_F$), equalizing condition contributions while maintaining accurate, unweighted $R^2X$ reporting.

### Inline Slice-Normalized Compression

```python
pf2_output, r2x = parafac2_nd(
    adata,
    rank=15,
    compress="auto",
    normalize_slices=True,
    random_state=42,
)
```

### Pre-Compressed Slice Normalization

When using `compress_dataset`, set `normalize_slices=True`. Slice weights are precalculated and cached in `compressed.slice_weights`:

```python
# Precalculate slice weights during compression
compressed = compress_dataset(
    adata,
    L=50,
    normalize_slices=True,
    random_state=42,
)

# Subsequent fits automatically use the precomputed slice weights
for r in [5, 10, 15]:
    output, r2x = parafac2_nd(compressed, rank=r, random_state=42)
    print(f"Rank {r} (normalized ALS) -> R2X: {r2x:.4f}")
```

---

## Approach 5: Hardware Acceleration (GPU / MLX / CuPy)

`parafac2` supports hardware acceleration on Apple Silicon (`mlx`) and NVIDIA GPUs (`cupy`). Compression projects the large data matrix via randomized SVD, which benefits significantly from GPU acceleration.

Specify the `backend` argument (`'mlx'`, `'cupy'`, `'cpu'`, or `None` for auto-detection):

```python
# Compress dataset using Apple Silicon GPU acceleration
compressed = compress_dataset(
    adata,
    L="auto",
    rank=30,
    backend="mlx",
    random_state=42,
)

# Fit on compressed cores
pf2_output, r2x = parafac2_nd(compressed, rank=20, random_state=42)
```

If no GPU backend is available or installed, `parafac2` falls back to CPU NumPy/SciPy computation automatically.

---

## Summary of Approaches

| Approach                         | Use Case                                    | Function Call                                                                    | Key Advantage                                     |
| :------------------------------- | :------------------------------------------ | :------------------------------------------------------------------------------- | :------------------------------------------------ |
| **One-Step Inline Auto**         | Single-rank fit with default parameters     | `parafac2_nd(adata, rank=R, compress="auto")`                                    | Simplest interface, optimal automatic dimensions  |
| **One-Step Inline Custom**       | Custom gene/cell subspace bounds            | `parafac2_nd(adata, rank=R, compress=(L_g, L_c))`                                | Fine-grained control over subspace truncation     |
| **Pre-Compressed Rank Sweep**    | Exploring multiple ranks ($R^2X$ curve)     | `compressed = compress_dataset(adata, ...)`<br>`parafac2_nd(compressed, rank=r)` | Raw matrix touched only once; instant rank fits   |
| **Gene-Only Compression**        | Small cell counts or exact cell coordinates | `compress=(L_g, None)` or `L=(L_g, None)`                                        | Skips per-condition SVDs; fast compression        |
| **Slice-Normalized Compression** | Imbalanced sample cell counts or variance   | `normalize_slices=True`                                                          | Prevents large conditions from dominating factors |
| **Hardware Accelerated**         | Large single-cell matrices on GPU / Mac     | `backend="mlx"` or `backend="cupy"`                                              | Accelerated randomized SVD and matrix products    |

---

## Complete End-to-End Workflow Example

Here is a complete, self-contained script demonstrating pre-compression, a rank sweep, selecting the best model, and storing results:

```python
import anndata
import numpy as np
from scipy.sparse import csr_array
from parafac2 import compress_dataset, parafac2_nd, store_pf2
from parafac2.normalize import prepare_dataset

# 1. Load or prepare single-cell dataset
# adata = prepare_dataset(raw_adata, condition_name="condition", geneThreshold=0.01)

# Synthetic demo data:
rng = np.random.default_rng(0)
n_cells, n_genes, n_conds = 3000, 1000, 6
mat = (rng.random((n_cells, n_genes)) < 0.05).astype(np.float32) * rng.exponential(1.0, size=(n_cells, n_genes)).astype(np.float32)
obs = {"condition_unique_idxs": np.repeat(np.arange(n_conds), n_cells // n_conds)}
var = {"means": np.ravel(mat.mean(axis=0))}
adata = anndata.AnnData(X=csr_array(mat), obs=obs, var=var)

# 2. Compress once with capacity for up to rank 30
compressed = compress_dataset(
    adata,
    L="auto",
    rank=30,
    normalize_slices=True,
    random_state=42,
)

print(f"Compressed dataset: L_g={compressed.L_g}, max_L_c={compressed.max_cell_dim}")
print(f"Variance retained: {1.0 - compressed.lost_var / compressed.norm_tensor:.2%}")

# 3. Fast rank sweep
ranks = [5, 10, 15, 20, 25, 30]
models = {}
for r in ranks:
    output, r2x = parafac2_nd(compressed, rank=r, tol=1e-6, random_state=42)
    models[r] = output
    print(f"Rank {r:2d} -> R2X = {r2x:.4f}")

# 4. Store selected rank into AnnData
best_rank = 15
adata = store_pf2(compressed, models[best_rank])

# 5. Access outputs in AnnData
print("\nStored outputs in AnnData:")
print("- Component weights (Pf2_weights):", adata.uns["Pf2_weights"].shape)
print("- Condition factors (Pf2_A):", adata.uns["Pf2_A"].shape)
print("- Eigen-state factors (Pf2_B):", adata.uns["Pf2_B"].shape)
print("- Gene factors (Pf2_C):", adata.varm["Pf2_C"].shape)
print("- Cell projections (projections):", adata.obsm["projections"].shape)
print("- Weighted cell projections:", adata.obsm["weighted_projections"].shape)
```
