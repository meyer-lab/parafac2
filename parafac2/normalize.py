from typing import cast

import anndata
import numpy as np
from scipy.sparse import csr_array, issparse


def prepare_dataset(
    X: anndata.AnnData, condition_name: str, geneThreshold: float
) -> anndata.AnnData:
    assert issparse(X.X)
    X.X = csr_array(X.X)
    X_X = cast("csr_array", X.X)
    assert np.amin(X_X.data) >= 0.0

    # Convert to float32 before filtering and copy to halve peak memory
    if X_X.dtype != np.float32:
        X_X.data = X_X.data.astype(np.float32)

    # Filter out genes with too few reads, and cells with fewer than 10 counts
    X = X[X_X.sum(axis=1) > 10, X_X.mean(axis=0) > geneThreshold]

    # Copy so that the subsetting is preserved
    X._init_as_actual(X.copy())
    X.X = csr_array(X.X)
    X_X = cast("csr_array", X.X)

    ## Normalize total counts per cell
    # Keep the counts on a reasonable scale to avoid accuracy issues
    counts_per_cell = X_X.sum(axis=1)
    counts_per_cell /= np.median(counts_per_cell)
    # inplace csr row scale
    X_X.data /= np.repeat(counts_per_cell, np.diff(X_X.indptr))

    # Scale genes by sum, inplace csr col scale
    X_X.data /= X_X.sum(axis=0).take(X_X.indices, mode="clip")

    # Transform values in-place to avoid nnz-sized temporaries
    X_X.data *= np.float32(1000.0)
    X_X.data += np.float32(1.0)
    np.log10(X_X.data, out=X_X.data)

    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex

    # Pre-calculate gene means
    X.var["means"] = X_X.mean(axis=0)

    return X
