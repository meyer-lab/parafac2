import numpy as np
import anndata
from scipy.sparse import csc_matrix, csr_matrix, csr_array
from sklearn.utils.sparsefuncs import (
    inplace_column_scale,
    mean_variance_axis,
    inplace_row_scale,
)


def normalize_total(adata: anndata.AnnData):
    counts_per_cell = np.array(adata.X.sum(axis=1)).flatten()  # type: ignore
    cell_subset = counts_per_cell > 0

    if issubclass(adata.X.dtype.type, (int, np.integer)):  # type: ignore
        adata.X = adata.X.astype(np.float32)  # type: ignore

    counts_per_cell /= np.median(counts_per_cell[cell_subset])

    inplace_row_scale(adata.X, 1.0 / np.clip(counts_per_cell, 1e-12, None))  # type: ignore


def prepare_dataset(
    X: anndata.AnnData, condition_name: str, geneThreshold: float
) -> anndata.AnnData:
    assert isinstance(X.X, csc_matrix | csr_matrix)
    assert np.amin(X.X.data) >= 0.0

    # Filter out genes with too few reads
    readmean, _ = mean_variance_axis(X.X, axis=0)  # type: ignore
    X = X[:, readmean > geneThreshold]

    # Copy so that the subsetting is preserved
    X._init_as_actual(X.copy())

    # Normalize read depth
    normalize_total(X)

    # Scale genes by sum
    readmean, _ = mean_variance_axis(X.X, axis=0)  # type: ignore
    readsum = X.shape[0] * readmean
    inplace_column_scale(X.X, 1.0 / readsum)  # type: ignore

    # Transform values
    X.X.data = np.log10((1000.0 * X.X.data) + 1.0)  # type: ignore

    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex

    # Pre-calculate gene means
    means, _ = mean_variance_axis(X.X, axis=0)  # type: ignore
    X.var["means"] = means

    return X


def deviance_prepare_dataset(
    X: anndata.AnnData, condition_name: str, geneThreshold: float
) -> anndata.AnnData:
    X.X = csr_array(X.X)  # type: ignore
    assert np.amin(X.X.data) >= 0.0

    # Filter out genes with too few reads
    X = X[:, X.X.mean(axis=0) > geneThreshold]

    # Copy so that the subsetting is preserved
    X._init_as_actual(X.copy())

    # deviance transform
    y_ij = X.X.toarray()  # type: ignore

    # counts per cell
    n_i = y_ij.sum(axis=1)

    # MLE of gene expression
    pi_j = y_ij.sum(axis=0) / np.sum(n_i)

    non_y_ij = n_i[:, None] - y_ij
    mu_ij = n_i[:, None] * pi_j[None, :]
    signs = np.sign(y_ij - pi_j[None, :])

    first_term = 2 * y_ij * np.log(np.maximum(y_ij, 1.0) / mu_ij)
    second_term = 2 * non_y_ij * np.log(non_y_ij / (n_i[:, None] - mu_ij))

    X.X = signs * np.sqrt(np.maximum(first_term + second_term, 0.0))
    assert np.all(np.isfinite(X.X))  # type: ignore

    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex

    # Skip gene means
    X.var["means"] = np.zeros(X.shape[1])

    return X
