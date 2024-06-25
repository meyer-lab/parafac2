import numpy as np
from ..jointdiag import jointdiag


def SyntheticData(k: int, d: int):
    """
    Generates random diagonal tensor, and random mixing matrix
    Multiplies every slice of diagonal tensor with matrix 'synthetic' S * D * S^-1
    Sends altered tensor into jointdiag function
    Returs diagonal tensor estimate and mixing matrix estimate
    """
    rng = np.random.RandomState()
    mixing = rng.randn(d, d)
    diags = np.zeros((d, d, k))
    synthetic = np.zeros((d, d, k))

    for i in range(k):
        temp_diag = np.diag(rng.randn(d))
        diags[:, :, i] = temp_diag
        synthetic[:, :, i] = np.linalg.inv(mixing) @ temp_diag @ mixing

    diag_est, mixing_est = jointdiag(synthetic, verbose=False)
    return diags, diag_est, mixing, mixing_est, synthetic


def test_jointdiag():
    diags, diag_est, _, _, _ = SyntheticData(40, 22)

    ## Sorts outputted diagonal data
    idx_est = np.argsort(np.diag(diag_est[:, :, 0]))
    idx = np.argsort(np.diag(diags[:, :, 0]))
    diag_est = diag_est[idx_est, idx_est, :]
    diags = diags[idx, idx, :]

    np.testing.assert_allclose(diags, diag_est, atol=1e-10)
