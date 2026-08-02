from typing import cast

import numpy as np
from scipy.sparse import csr_matrix, issparse


class SampleArray:
    """
    Wrapper for a single sample matrix (csr_matrix or np.ndarray) and its gene means.
    Automatically performs mean-centering during left and right matrix multiplications.
    """

    __array_priority__ = 1000

    def __init__(self, mat: np.ndarray | csr_matrix, means: np.ndarray):
        if issparse(mat):
            self.mat = csr_matrix(mat)
        else:
            self.mat = np.asarray(mat)
        self.means = np.asarray(means).ravel()

    @property
    def shape(self) -> tuple[int, int]:
        return self.mat.shape

    @property
    def dtype(self):
        return self.mat.dtype

    @property
    def ndim(self) -> int:
        return 2

    def __len__(self) -> int:
        return self.mat.shape[0]

    def toarray(self) -> np.ndarray:
        """Return the dense, mean-centered matrix."""
        dense = (
            cast("csr_matrix", self.mat).toarray()
            if issparse(self.mat)
            else self.mat
        )
        return dense - self.means

    def norm_sq(self) -> float:
        """Return the squared Frobenius norm of the mean-centered matrix."""
        return float(np.sum(self.toarray() ** 2))

    def __matmul__(self, rhs: np.ndarray) -> np.ndarray:
        """
        Left matrix multiplication: self @ rhs
        Computes (self.mat - means) @ rhs = self.mat @ rhs - means @ rhs
        """
        res = self.mat @ rhs
        res_arr = res.toarray() if issparse(res) else np.asarray(res)
        return res_arr - (self.means @ rhs)

    def __rmatmul__(self, lhs: np.ndarray) -> np.ndarray:
        """
        Right matrix multiplication: lhs @ self
        Computes lhs @ (self.mat - means) = lhs @ self.mat -
        outer(sum(lhs, axis=1), means)
        """
        res = lhs @ self.mat
        res_arr = res.toarray() if issparse(res) else np.asarray(res)
        if lhs.ndim == 2:
            row_sums = np.sum(lhs, axis=1)
            return res_arr - np.outer(row_sums, self.means)
        else:
            return res_arr - np.sum(lhs) * self.means


# Alias for snake_case class reference
sample_array = SampleArray
