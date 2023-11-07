""" Partial Least Squares for PARAFAC2 """

import numpy as np
from numpy.linalg import norm, svd
from copy import deepcopy


def pf2_pls(Xs, Y, rank=1, max_iter=100, epsilon=1e-7):
    # Sanity check
    assert len(Xs) > 1
    K = len(Xs)
    assert Y.shape[0] == K
    J = Xs[0].shape[0]
    for k in range(K):
        assert Xs[k].shape[0] == J
    Ns = [X.shape[1] for X in Xs]
    oXs = deepcopy(Xs)
    Xs = deepcopy(Xs)
    Y = deepcopy(Y)

    # Initialization
    T = np.ones((K, rank))
    Fk = [np.zeros((n, rank)) for n in Ns]
    A = np.ones((J, rank)) / np.sqrt(J)

    for n_comp in range(rank):
        for n_iter in range(max_iter):
            old_a = np.copy(A[:, n_comp])
            for k in range(K):
                Fk[k][:, n_comp] = Y[k] * Xs[k].T @ A[:, n_comp]
                Fk[k][:, n_comp] = Fk[k][:, n_comp] / norm(Fk[k][:, n_comp])
            A[:, n_comp] = np.array([Xs[k] @ Fk[k][:, n_comp] for k in range(K)]).T @ Y
            A[:, n_comp] = A[:, n_comp] / norm(A[:, n_comp])
            if norm(old_a - A[:, n_comp]) < epsilon:
                break
        for k in range(K):
            T[k, n_comp] = A[:, n_comp].T @ Xs[k] @ Fk[k][:, n_comp].T
        for k in range(K):
            Xs[k] = oXs[k] - T[k, n_comp] * A[:, :(n_comp+1)] @ Fk[k][:, :(n_comp+1)].T
    return T, A, Fk
