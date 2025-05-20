import anndata
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.utils.sparsefuncs import (
    inplace_column_scale,
    inplace_row_scale,
    mean_variance_axis,
)


def normalize_total(adata: anndata.AnnData):
    counts_per_cell = np.array(adata.X.sum(axis=1)).flatten()  # type: ignore
    cell_subset = counts_per_cell > 0

    if issubclass(adata.X.dtype.type, int | np.integer):  # type: ignore
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





def prepare_dataset(X: AnnData) -> AnnData:
    """Prepare the AnnData dataset for normalization.

    Args:
        X (AnnData): The input AnnData object.

    Returns:
        AnnData: The prepared AnnData object.

    """
    X.X = csr_array(X.X)  # type: ignore
    assert np.amin(X.X.data) >= 0.0

    # Remove cells and genes with fewer than 20 reads
    X = X[X.X.sum(axis=1) > 20, X.X.sum(axis=0) > 20]

    # Copy so that the subsetting is preserved
    X._init_as_actual(X.copy())

    # counts per gene
    X.var["p_i"] = X.X.sum(axis=0)

    return X



    def get_dense(self) -> np.ndarray:
        # Convert to dense - consider memory implications
        y_ij = self.data.toarray()

        # Counts per cell (ensure 1D array)
        n_i = self.data.sum(axis=1)

        # Ensure n_i is broadcastable as a column vector
        n_i_col = n_i.reshape(-1, 1)

        mu_ij = n_i_col * self.pi_j

        # --- Calculate Deviance Terms using numerically stable xlogy ---
        # D = 2 * [ y*log(y/mu) + (n-y)*log((n-y)/(n-mu)) ]
        # D = 2 * [ (xlogy(y, y) - xlogy(y, mu)) + (xlogy(n-y, n-y) - xlogy(n-y, n-mu)) ]

        n_minus_y = n_i_col - y_ij
        n_minus_mu = n_i_col - mu_ij

        # Term 1: y * log(y / mu) = xlogy(y, y) - xlogy(y, mu)
        # xlogy handles y=0 case correctly returning 0.
        row, col = self.data.nonzero()
        mu_ij_nn = n_i_col[row, 0] * self.pi_j[0, col]
        term1 = self.data.data * np.log(self.data.data / mu_ij_nn)

        # Term 2: (n-y) * log((n-y) / (n-mu)) = xlogy(n-y, n-y) - xlogy(n-y, n-mu)
        # xlogy handles n-y=0 case correctly returning 0.
        # This corresponds to the second term of the deviance formula:
        # (n-y) * log((n-y) / (n-mu)) = xlogy(n-y, n-y) - xlogy(n-y, n-mu)
        deviance = xlogy(n_minus_y, n_minus_y / n_minus_mu)
        deviance[row, col] += term1

        # Calculate full deviance: D = 2 * (term1 + term2)
        # Handle potential floating point inaccuracies leading to small negatives
        deviance = 2 * deviance
        deviance = np.maximum(deviance, 0.0)  # Ensure non-negative before sqrt

        # Calculate signed square root residuals: sign(y - mu) * sqrt(D)
        signs = np.sign(y_ij - mu_ij)

        return signs * np.sqrt(deviance)








