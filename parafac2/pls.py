""" Partial Least Squares for PARAFAC2 """

import numpy as np
from numpy.linalg import norm, lstsq
from copy import deepcopy


def calcR2X(Xorig, Xrecon):
    return 1 - norm(Xorig - Xrecon) ** 2 / norm(Xorig) ** 2


class pf2PLS():
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def preprocess(self, Xs, Y):
        assert len(Xs) > 1
        self.K = len(Xs)
        assert Y.shape[0] == self.K
        self.J = Xs[0].shape[0]
        for k in range(self.K):
            assert Xs[k].shape[0] == self.J
        self.Ns = [X.shape[1] for X in Xs]

        self.X_means = [np.mean(X) for X in Xs]
        self.Y_mean = np.mean(Y, axis=0)
        for k in range(self.K):
            Xs[k] = Xs[k] - self.X_means[k]
        Y = Y - self.Y_mean
        self.oXs = deepcopy(Xs)
        self.oY = deepcopy(Y)
        return deepcopy(Xs), deepcopy(Y)

    def initialize(self):
        self.T = np.zeros((self.K, self.rank))
        self.Fk = [np.ones((n, self.rank)) for n in self.Ns]
        self.A = np.ones((self.J, self.rank)) / np.sqrt(self.J)
        self.B = np.zeros((self.rank, self.rank))
        self.Xfacs = [self.T, self.A, self.Fk]

        self.R2Xs = np.zeros((self.rank, self.K))
        self.R2Ys = np.zeros((self.rank))

    def fit(self, Xs, Y, max_iter=1000, epsilon=1e-7):
        Xs, Y = self.preprocess(Xs, Y)
        self.initialize()
        for n_comp in range(self.rank):
            for n_iter in range(max_iter):
                old_a = np.copy(self.A[:, n_comp])
                for k in range(self.K):
                    self.Fk[k][:, n_comp] = Y[k] * Xs[k].T @ self.A[:, n_comp]
                    self.Fk[k][:, n_comp] = self.Fk[k][:, n_comp] / norm(self.Fk[k][:, n_comp])
                self.A[:, n_comp] = np.array([Xs[k] @ self.Fk[k][:, n_comp] for k in range(self.K)]).T @ Y
                self.A[:, n_comp] = self.A[:, n_comp] / norm(self.A[:, n_comp])
                if norm(old_a - self.A[:, n_comp]) < epsilon:
                    break
            for k in range(self.K):
                self.T[k, n_comp] = self.A[:, n_comp].T @ Xs[k] @ self.Fk[k][:, n_comp].T
                Xs[k] -= self.T[k, n_comp] * self.A[:, [n_comp]] @ self.Fk[k][:, [n_comp]].T

            self.B[:, n_comp] = lstsq(self.T, Y, rcond=-1)[0]
            Y -= self.T @ self.B[:, n_comp]

            # Diagnosis
            for k in range(self.K):
                Xrecon = self.A[:, :(n_comp + 1)] * self.T[k, :(n_comp + 1)] @ self.Fk[k][:, :(n_comp + 1)].T
                self.R2Xs[n_comp, k] = calcR2X(self.oXs[k], Xrecon)
            self.R2Ys[n_comp] = calcR2X(self.oY, self.T @ self.B.T @ np.ones((self.rank)))
