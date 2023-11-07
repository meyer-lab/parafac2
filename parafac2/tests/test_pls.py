""" Test PLS for PARAFAC2 """

import numpy as np
from ..pls import *


def genSim(Ns = [7, 5, 9, 3, 6, 12], J = 4):
    K = len(Ns)
    Y = np.random.rand(K)
    Y -= np.mean(Y)

    Xs = [np.random.rand(J, Ns[k]) for k in range(K)]
    for k in range(K):
        Xs[k] = Xs[k] - np.mean(Xs[k])
    return Xs, Y


def test_pf2PLS():
    Xs, Y = genSim()
    pls = pf2PLS(5)
    pls.fit(Xs, Y)
    assert np.all(np.diff(pls.R2Xs, axis=0) >=0)
    #assert np.all(np.diff(pls.R2Ys) >=0)
