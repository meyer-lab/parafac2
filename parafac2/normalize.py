import anndata
import numpy as np
from scipy.sparse import csr_array, issparse
from scipy.special import xlogy


def prepare_dataset(
    X: anndata.AnnData, condition_name: str, geneThreshold: float, deviance=False
) -> anndata.AnnData:
    assert issparse(X.X)
    X.X = csr_array(X.X)
    assert np.amin(X.X.data) >= 0.0

    # Filter out genes with too few reads, and cells with fewer than 10 counts
    X = X[X.X.sum(axis=1) > 10, X.X.mean(axis=0) > geneThreshold]

    # Copy so that the subsetting is preserved
    X._init_as_actual(X.copy())
    X.X = csr_array(X.X)

    # Convert counts to floats
    if issubclass(X.X.dtype.type, int | np.integer):
        X.X.data = X.X.data.astype(np.float32)

    if deviance:
        X.X = get_deviance(X.X)
    else:
        ## Normalize total counts per cell
        # Keep the counts on a reasonable scale to avoid accuracy issues
        counts_per_cell = X.X.sum(axis=1)
        counts_per_cell /= np.median(counts_per_cell)
        # inplace csr row scale
        X.X.data /= np.repeat(counts_per_cell, np.diff(X.X.indptr))

        # Scale genes by sum, inplace csr col scale
        X.X.data /= X.X.sum(axis=0).take(X.X.indices, mode="clip")

        # Transform values
        X.X.data = np.log10((1000.0 * X.X.data) + 1.0)

    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex

    # Pre-calculate gene means
    X.var["means"] = X.X.mean(axis=0)

    return X


def get_deviance(data: csr_array) -> np.ndarray:
    """
    Calculate signed square root deviance residuals for a sparse count matrix.
    This function computes deviance residuals for count data assuming a binomial-like
    model where each cell has n_i total counts and each gene j has probability pi_j.
    The deviance residuals are calculated as sign(y - mu) * sqrt(D), where D is the
    deviance contribution from each observation.
    Parameters
    ----------
    data : csr_array
        Sparse matrix of count data with shape (n_cells, n_genes).
        Each entry y_ij represents the count for gene j in cell i.
    Returns
    -------
    np.ndarray
        Dense array of signed square root deviance residuals with the same shape
        as the input data. Positive values indicate observed counts greater than
        expected, negative values indicate observed counts less than expected.
    Notes
    -----
    The deviance is calculated using the formula:
    D = 2 * [y*log(y/mu) + (n-y)*log((n-y)/(n-mu))]
    Where:
    - y_ij is the observed count for gene j in cell i
    - mu_ij = n_i * pi_j is the expected count
    - n_i is the total count for cell i
    - pi_j is the proportion of total counts for gene j
    The function uses numerically stable implementations to handle edge cases
    where y=0 or n-y=0, and ensures the final deviance values are non-negative
    before taking the square root.
    """
    # merge duplicate entries in the sparse matrix by summing their values
    data.sum_duplicates()
    data.eliminate_zeros()

    # counts per gene
    pi_j = data.sum(axis=0)

    # Convert to dense - consider memory implications
    y_ij = data.toarray()

    # Counts per cell (ensure 1D array)
    n_i = data.sum(axis=1)

    # Ensure n_i is broadcastable as a column vector
    n_i_col = n_i.reshape(-1, 1)

    mu_ij = n_i_col * pi_j

    # --- Calculate Deviance Terms using numerically stable xlogy ---
    # D = 2 * [ y*log(y/mu) + (n-y)*log((n-y)/(n-mu)) ]
    # D = 2 * [ (xlogy(y, y) - xlogy(y, mu)) + (xlogy(n-y, n-y) - xlogy(n-y, n-mu)) ]

    n_minus_y = n_i_col - y_ij
    n_minus_mu = n_i_col - mu_ij

    # Term 1: y * log(y / mu) = xlogy(y, y) - xlogy(y, mu)
    # xlogy handles y=0 case correctly returning 0.
    row, col = data.nonzero()
    mu_ij_nn = n_i[row] * pi_j[col]
    term1 = data.data * np.log(data.data / mu_ij_nn)

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
    residuals = np.sqrt(deviance) * np.sign(y_ij - mu_ij)

    # z-score
    residuals -= np.mean(residuals, axis=0)
    residuals /= np.std(residuals, axis=0)
    return residuals
