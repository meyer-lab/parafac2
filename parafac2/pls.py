""" Partial Least Squares for PARAFAC2 """

import numpy as np
from numpy.linalg import norm, svd
from numpy import cov
from copy import deepcopy

def pf2_pls(Xs, Y, rank=2, max_iter=100):
    # Sanity check
    assert len(Xs) > 1
    K = len(Xs)
    assert Y.shape[0] == K
    J = Xs[0].shape[0]
    Ns = [X.shape[1] for X in Xs]
    oXs = deepcopy(Xs)

    # Initialization
    T = np.ones((K, rank))
    Fk = [np.zeros((n, rank)) for n in Ns]
    A = np.ones((J, rank))
    A[:, 0] = svd(np.hstack([X for X in Xs]))[0][:, 0]

    for n_comp in range(rank):
        for n_iter in range(max_iter):
            for k in range(K):
                Fk[k][:, n_comp] = Xs[k].T @ A[:, n_comp]
                Fk[k][:, n_comp] = Fk[k][:, n_comp] / norm(Fk[k][:, n_comp])
            A[:, n_comp] = np.array([Xs[k] @ Fk[k][:, n_comp] for k in range(K)]).T @ Y
            A[:, n_comp] = A[:, n_comp] / norm(A[:, n_comp])
            for k in range(K):
                T[k, n_comp] = A[:, n_comp].T @ Xs[k] @ Fk[k][:, n_comp].T
        for k in range(K):
            Xs[k] = oXs[k] - T[k, n_comp] * A[:, :n_comp] @ Fk[k][:, :n_comp].T
    return T, A, Fk
