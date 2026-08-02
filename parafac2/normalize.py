"""
Dataset preprocessing and normalization utilities for PARAFAC2 analysis.

This module provides functions to filter, normalize, and annotate single-cell
gene expression datasets stored in AnnData objects prior to PARAFAC2 matrix
factorization.
"""

import pandas as pd
from typing import cast

import anndata
import numpy as np
from scipy.sparse import csr_array, issparse


def prepare_dataset(
    X: anndata.AnnData, condition_name: str, geneThreshold: float
) -> anndata.AnnData:
    """Preprocess and normalize an AnnData dataset for PARAFAC2 factorization.

    Performs quality control filtering of low-count cells and low-expression
    genes, normalizes total cell counts and gene sums, applies a log10
    transformation, and computes metadata required by PARAFAC2 (condition
    indices and gene means).

    Parameters
    ----------
    X : anndata.AnnData
        Input single-cell dataset with raw count matrix stored in ``X.X``
        (must be a sparse matrix with non-negative values).
    condition_name : str
        Column name in ``X.obs`` identifying the sample or experimental
        condition grouping for each cell.
    geneThreshold : float
        Minimum threshold fraction for gene inclusion. Genes with total counts
        less than ``geneThreshold * total_cells`` are filtered out.

    Returns
    -------
    anndata.AnnData
        A filtered and normalized copy of the AnnData object. Contains the
        log-transformed normalized counts in ``X.X``, integer condition
        codes in ``X.obs["condition_unique_idxs"]``, and per-gene mean
        expression values in ``X.var["means"]``.
    """
    assert issparse(X.X)
    assert np.amin(X.X.data) >= 0.0

    # Filter out genes with too few reads, and cells with fewer than 10 counts
    cell_mask = np.ravel(X.X.sum(axis=1)) > 10
    gene_mask = np.ravel(X.X.sum(axis=0)) > (geneThreshold * X.X.shape[0])

    # Subset and materialize actual AnnData object before modifying X.X
    if cell_mask.all() and gene_mask.all():
        X = X.copy()
    else:
        X = X[cell_mask, gene_mask].copy()

    # Convert subset to csr_array and float32 data
    X.X = csr_array(X.X)
    X_X = cast("csr_array", X.X)

    if X_X.dtype != np.float32:
        X_X.data = X_X.data.astype(np.float32)

    ## Normalize total counts per cell
    # Keep the counts on a reasonable scale to avoid accuracy issues
    counts_per_cell = np.ravel(X_X.sum(axis=1)).astype(np.float32, copy=False)
    counts_per_cell /= np.median(counts_per_cell)
    # In-place CSR row scaling
    X_X.data /= np.repeat(counts_per_cell, np.diff(X_X.indptr))

    # Scale genes by sum, in-place CSR column scaling
    gene_sums = np.ravel(X_X.sum(axis=0)).astype(np.float32, copy=False)
    X_X.data /= gene_sums[X_X.indices]

    # Transform values in-place to avoid nnz-sized temporaries
    X_X.data *= np.float32(1000.0)
    X_X.data += np.float32(1.0)
    np.log10(X_X.data, out=X_X.data)

    # Get the indices for subsetting the data
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs[condition_name]).codes

    # Pre-calculate gene means
    X.var["means"] = np.ravel(X_X.mean(axis=0))

    return X
