"""Shared Hypothesis strategies for property-based tests.

Centralizes strategies for generating the small dense/sparse matrices and
condition-index arrays that the PARAFAC2 numerical routines operate on, kept
small and well-scaled so the resulting examples stay fast and numerically
well-behaved (finite, not pathologically ill-conditioned).
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.sparse import csr_array

# Keep shapes small: these routines are exercised end-to-end (SVDs, eigh,
# ALS sweeps), so large examples would make the suite slow without adding
# coverage of new code paths.
_DIM = st.integers(min_value=1, max_value=12)


def real_arrays(shape, dtype=np.float64, min_value=-10.0, max_value=10.0):
    """A strategy for finite, moderately scaled float arrays of ``shape``."""
    return arrays(
        dtype=dtype,
        shape=shape,
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
            width=32 if np.dtype(dtype).itemsize == 4 else 64,
        ),
    )


@st.composite
def dense_matrices(draw, min_rows=1, max_rows=12, min_cols=1, max_cols=12):
    """A strategy for small finite dense matrices with a random shape."""
    rows = draw(st.integers(min_rows, max_rows))
    cols = draw(st.integers(min_cols, max_cols))
    return draw(real_arrays((rows, cols)))


@st.composite
def dense_matrix_with_sparsity(draw, min_rows=2, max_rows=15, min_cols=2, max_cols=10):
    """A dense matrix with a random fraction of entries forced to zero.

    Returns the resulting (sparsified) dense array, so callers can build both
    a dense and an equivalent `csr_array` view of the same data.
    """
    mat = draw(dense_matrices(min_rows, max_rows, min_cols, max_cols))
    zero_mask = draw(arrays(dtype=bool, shape=mat.shape, elements=st.booleans()))
    return np.where(zero_mask, 0.0, mat)


@st.composite
def condition_groups(draw, n_rows, min_cond=1, max_cond=5):
    """A grouped (contiguous, sorted) condition-index array of length `n_rows`.

    Matches the layout `anndata.concat` produces: rows are grouped by
    condition and in ascending condition order.
    """
    n_cond = draw(st.integers(min_cond, min(max_cond, n_rows)))
    # Random composition of n_rows into n_cond positive parts.
    cuts = sorted(
        draw(
            st.lists(
                st.integers(1, n_rows - 1) if n_rows > 1 else st.just(1),
                min_size=n_cond - 1,
                max_size=n_cond - 1,
                unique=True,
            )
        )
    )
    bounds = [0, *cuts, n_rows]
    idxs = np.concatenate(
        [np.full(bounds[i + 1] - bounds[i], i) for i in range(n_cond)]
    )
    return idxs.astype(int)


def to_csr(mat: np.ndarray) -> csr_array:
    """Convert a dense array to a `csr_array`, matching the project's dtype use."""
    return csr_array(mat)
