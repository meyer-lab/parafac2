from copy import deepcopy
from typing import List, Literal

import numpy as np

"""
Parafac2 Partial Least Squares Regression (PF2-PLSR)

Derivation + explanation:
https://notability.com/n/tmoLhNfW2D8XafYQbXbLA
"""


class PF2_PLSR:
    def __init__(self, rank):
        self.rank = rank

    def fit(
        self,
        X: List[np.ndarray],
        y: np.ndarray,
        max_iter=1000,
        tol=1e-8,
        verbose=False,
        init: Literal["ones", "random"] = "ones",
        center=True,
    ):
        """
        Fits the data, X and y, to generate factors in X which correspond to
        maximum covariation in y.

        Args:
            X: List of slices of X along the I dimension. Each slice of X should
              be of size Ji x K, where Ji is free to vary across slices and K is
              shared.
            y: Array of labels of size I.
            max_iter: Maximum iterations.
            tol: Absolute tolerance in the change in all factors at which to
              terminate fit.
            verbose: Verbose
            init: Initialization for factors. Can be either "ones" or "random".
            center: Whether or not to center each X slice to 0 mean.

        Results from fit can be accessed through:

        (self.Omega_J, self.Omega_K, T) in the notation used in the document
          linked at the top of the file. Where Omega_J is of shape (rank, I, J),
          Omega_K (rank, K), and T (rank, I).

        OR

        (self.A, self.Bi, self.C) in the notation used in tensorly parafac2.
          Where A is of shape (I, rank), Bi is of shape (Ji, rank), and C is of shape
          (K, rank)

        """
        X, y, y_init, T, Omega_J, Omega_K = self.initialize_fit(X, y, center=center)

        for r in range(self.rank):
            W_J, w_K = initialize_w(X, init)
            Z = y[:, None, None] * X

            tol_reached = False
            for iter in range(max_iter):
                W_J_old = W_J
                w_K_old = w_K

                # solve for W_J
                psi_J = np.einsum("ijk,k->ij", Z, w_K)
                W_J = psi_J / np.linalg.norm(psi_J, axis=1)[:, None]

                # solve for w_K
                Psi_K = np.einsum("ijk,ij->k", Z, W_J)
                w_K = Psi_K / np.linalg.norm(Psi_K)

                if (
                    np.linalg.norm(W_J - W_J_old) < tol
                    and np.linalg.norm(w_K - w_K_old) < tol
                ):
                    tol_reached = True
                    if verbose:
                        print("Optimization tolerance reached")
                    break

            if not tol_reached and verbose:
                print("Max iterations reached")

            # store W_I and w_j
            Omega_J[r] = W_J
            Omega_K[r] = w_K

            # compute t by projecting X onto W_J and w_K
            t = np.einsum("ijk,ij,k->i", X, W_J, w_K)
            T[r] = t

            b = self.solve_regression_coefficients(T.T, y)

            # deflate X
            X -= np.einsum("i,ij,k->ijk", t, W_J, w_K)

            # deflate Y:
            y -= T.T @ b

        # compute final regression coefficients with all X components against original y
        b = self.solve_regression_coefficients(T.T, y_init)

        # assign member variables
        self.store_fit_results(Omega_J, Omega_K, T, b)

        return self

    def initialize_fit(self, X: List[np.ndarray], y: np.ndarray, center=True):
        """Initialize factor arrays and preprocess X and y."""
        X, y = deepcopy(X), deepcopy(y)
        Ks = np.array([X_slice.shape[1] for X_slice in X])
        assert np.all(
            Ks == Ks[0]
        ), "The second (K) dimension in each X slice should be equal"

        if center:
            # center X and y
            X = center_X(X)
            y -= np.mean(y)

        # concatenate slices into tensor and store the lengths of the
        # unaligned dimension
        self.Ji = np.array([X_slice.shape[0] for X_slice in X])
        X = X_slices_to_tensor(X)

        assert X.ndim == 3, "X must be 3d tensor"
        assert y.ndim == 1, "y must be 1d tensor"
        self.I, self.J, self.K = X.shape
        assert y.shape == (self.I,)

        y_init = np.copy(y)
        T = np.zeros((self.rank, self.I), dtype=float)
        if not center:
            T = np.vstack((T, np.ones(self.I)))  # add 1s for bias term if not centering
        Omega_J = np.zeros((self.rank, self.I, self.J), dtype=float)
        Omega_K = np.zeros((self.rank, self.K), dtype=float)
        return X, y, y_init, T, Omega_J, Omega_K

    def store_fit_results(
        self, Omega_J: np.ndarray, Omega_K: np.ndarray, T: np.ndarray, b: np.ndarray
    ):
        self.Omega_J, self.Omega_K, self.T, self.b = Omega_J, Omega_K, T, b

        # store y predictions
        self.y_fit = T.T @ b

        # make sure Omega_J entries are zero where the corresponding slice in X ends
        for i in range(Omega_J.shape[1]):
            Omega_J[:, i, self.Ji[i] :] = 0

        # store results in pf2 notation
        self.A = T.T
        self.Bi = []
        for i in range(Omega_J.shape[1]):
            self.Bi.append(Omega_J[:, i, : self.Ji[i]].T)
        self.C = Omega_K.T

    def reconstruct_X(self) -> np.ndarray:
        """
        Reconstructs X from fitted factors.

        Returns: X as a np.ndarray. Slices are padded with zeros if needed.
        """
        self.assert_fit_has_run()
        return reconstruct_X(self.Omega_J, self.Omega_K, self.T)

    def assert_fit_has_run(self):
        assert hasattr(
            self, "Omega_J"
        ), "fit() must be called before accessing this functionality"

    def solve_regression_coefficients(
        self, predictors: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Solve the linear regression problem and return regression coefficients.

        Args:
            predictors: array of shape (observations, variables).
            y: array of shape (observations,)

        Returns: linear regression coefficients of shape (variables,)
        """
        return np.linalg.lstsq(predictors, y, rcond=-1)[0]


def initialize_w(X: np.ndarray, init: Literal):
    I, J, K = X.shape
    if init == "random":
        W_J = np.random.uniform(-1, 1, (I, J))
        w_K = np.random.uniform(-1, 1, K)
    if init == "ones":
        W_J = np.ones((I, J))
        w_K = np.ones(K)
    W_J /= np.linalg.norm(W_J, axis=1)[:, None]
    w_K /= np.linalg.norm(w_K)
    return W_J, w_K


def reconstruct_X(
    Omega_J: np.ndarray, Omega_K: np.ndarray, T: np.ndarray
) -> np.ndarray:
    if T.shape[0] == Omega_J.shape[0] + 1:
        T = T[:-1]  # last row is 1s for bias term
    return np.einsum("ri,rij,rk->ijk", T, Omega_J, Omega_K)


def center_X(X: List):
    return [X_slice - np.mean(X_slice) for X_slice in X]


def X_slices_to_tensor(X: List):
    """Converts X slices to tensor, padding shorter slices with 0s."""
    X = deepcopy(X)
    Ji = np.array([X_slice.shape[0] for X_slice in X])
    J = np.max(Ji)
    return np.array(
        [np.pad(X_slice, ((0, J - X_slice.shape[0]), (0, 0))) for X_slice in X]
    )


def X_tensor_to_slices(X: np.ndarray, Ji: np.ndarray = None):
    """Converts X tensor to slices. If Ji is supplied, each slice along the I
    dimension of X will be be indexed as X[i][:Ji]"""
    X = np.copy(X)
    if Ji is None:
        Ji = np.full(len(X), X.shape[1])
    return [X[i, : Ji[i]] for i in range(X.shape[0])]
