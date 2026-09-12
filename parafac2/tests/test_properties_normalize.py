"""Property-based tests for ``normalize.prepare_dataset``."""

from __future__ import annotations

from typing import cast

import anndata
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import sparse as sps
from scipy.sparse import csr_array

from ..normalize import prepare_dataset


@st.composite
def count_matrices(draw, min_rows=5, max_rows=25, min_cols=5, max_cols=15):
    """A small non-negative "count" matrix, as `prepare_dataset` expects.

    Row 0 and column 0 are boosted so at least one cell and one gene always
    survives `prepare_dataset`'s count filters, regardless of what the rest
    of the (still arbitrary, possibly all-zero) matrix looks like: the
    all-filtered-out case crashes with an opaque ZeroDivisionError, tracked
    separately by `test_prepare_dataset_raises_a_clear_error_when_everything_is_filtered`.
    """
    rows = draw(st.integers(min_rows, max_rows))
    cols = draw(st.integers(min_cols, max_cols))
    counts = draw(
        arrays(
            dtype=np.float32,
            shape=(rows, cols),
            elements=st.floats(
                min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    counts[0, :] += 1000.0
    counts[:, 0] += 1000.0
    return counts


@given(
    counts=count_matrices(),
    gene_threshold=st.floats(min_value=0.0, max_value=0.2),
    n_conditions=st.integers(1, 4),
    data=st.data(),
)
@settings(max_examples=40)
def test_prepare_dataset_invariants(counts, gene_threshold, n_conditions, data):
    """Regardless of the random count matrix and condition labeling,
    prepare_dataset must produce a self-consistent, well-formed output."""
    adata = anndata.AnnData(sps.csr_array(counts))
    labels = data.draw(
        st.lists(
            st.integers(0, n_conditions - 1),
            min_size=counts.shape[0],
            max_size=counts.shape[0],
        )
    )
    adata.obs["condition"] = [f"cond_{i}" for i in labels]

    out = prepare_dataset(adata, "condition", gene_threshold)
    out_X = cast("csr_array", out.X)

    # Shape can only shrink (rows/genes are filtered, never added).
    assert out.shape[0] <= counts.shape[0]
    assert out.shape[1] <= counts.shape[1]

    # Dtype is standardized to float32.
    assert out_X.dtype == np.float32

    # Metadata is present and internally consistent.
    assert "condition_unique_idxs" in out.obs
    assert "means" in out.var
    assert len(out.var["means"]) == out.shape[1]
    assert set(out.obs["condition_unique_idxs"].unique()).issubset(
        set(range(n_conditions))
    )

    # Values are finite: no filtered-out zero row/column should have produced
    # a division by zero that silently propagated as inf/nan.
    assert np.all(np.isfinite(out_X.data))
    assert np.all(np.isfinite(out.var["means"]))


def test_prepare_dataset_raises_a_clear_error_when_everything_is_filtered():
    """Regression test for a fixed bug: prepare_dataset used to crash with
    an opaque `ZeroDivisionError` (from deep inside scipy.sparse's `.mean()`)
    instead of a clear, actionable error when the cell/gene count filters
    removed every row (e.g. every cell's total count is <= 10) -- a
    plausible real input (a low-depth/heavily subsampled dataset, or a
    small toy/test dataset). Fixed by validating that at least one cell and
    one gene survive the filters before proceeding.
    """
    counts = np.ones(
        (5, 5), dtype=np.float32
    )  # every row sums to 5, below the >10 filter
    adata = anndata.AnnData(sps.csr_array(counts))
    adata.obs["condition"] = ["c"] * 5

    with pytest.raises(ValueError, match="no cells|no genes|filter"):
        prepare_dataset(adata, "condition", 0.0)


def test_prepare_dataset_raises_when_no_gene_passes_the_threshold():
    """Cells survive the count filter but every gene is below geneThreshold."""
    counts = np.ones((5, 5), dtype=np.float32) * 100.0  # cells easily pass
    adata = anndata.AnnData(sps.csr_array(counts))
    adata.obs["condition"] = ["c"] * 5

    with pytest.raises(ValueError, match="no genes"):
        prepare_dataset(adata, "condition", geneThreshold=1e6)
