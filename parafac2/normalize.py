import anndata
import numpy as np
from scipy.sparse import csr_array, issparse


def normalize_total(XX: csr_array) -> csr_array:
    """
    Normalize a sparse matrix, dividing each row by the median-normalized
    sum of its elements.
    """
    if issubclass(XX.dtype.type, int | np.integer):
        XX.data = XX.data.astype(np.float32)

    counts_per_cell = XX.sum(axis=1)

    # Keep the counts on a reasonable scale to avoid accuracy issues
    print(counts_per_cell)
    counts_per_cell /= np.median(counts_per_cell)

    XX /= counts_per_cell[:, None]

    return XX


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

    # Normalize read depth
    X.X = normalize_total(X.X)  # type: ignore

    # Scale genes by sum
    X.X /= X.X.sum(axis=0)[None, :]

    # Transform values
    X.X.data = np.log10((1000.0 * X.X.data) + 1.0)

    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)
    X.obs["condition_unique_idxs"] = sgIndex

    # Pre-calculate gene means
    X.var["means"] = X.X.mean(axis=0)

    return X
