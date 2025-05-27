import anndata
import numpy as np
from scipy.sparse import csr_array, issparse


def prepare_dataset(
    X: anndata.AnnData, condition_name: str, geneThreshold: float
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
