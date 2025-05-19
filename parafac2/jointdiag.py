from itertools import combinations

import numpy as np


def jointdiag(
    SMD: np.ndarray,
    MaxIter: int = 50,
    threshold: float = 1e-10,
    verbose=False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jointly diagonalizes n matrices, organized in tensor of dimension (k,k,n).
    Returns Diagonalized matrices.
    If showQ = True, returns transform matrix in second index.
    If showError = True, returns estimate of error in third index.
    """

    X = SMD.copy()
    D = X.shape[0]  # Dimension of square matrix slices
    assert X.ndim == 3, "Input must be a 3D tensor"
    assert X.shape[1] == D, "All slices must be square"
    assert np.all(np.isreal(X)), "Must be real-valued"

    # Initial error calculation
    # Transpose is because np.tril operates on the last two dimensions
    e = (
        np.linalg.norm(X) ** 2.0
        - np.linalg.norm(np.diagonal(X, axis1=1, axis2=2)) ** 2.0
    )

    if verbose:
        print(f"Sweep # 0: e = {e:.3e}")

    # Additional output parameters
    Q_total = np.eye(D)

    for k in range(MaxIter):
        # loop over all pairs of slices
        for p, q in combinations(range(D), 2):
            # Finds matrix slice with greatest variability among diagonal elements
            d_ = X[p, p, :] - X[q, q, :]
            h = np.argmax(np.abs(d_))

            # List of indices
            all_but_pq = list(set(range(D)) - set([p, q]))

            # Compute certain quantities
            dh = d_[h]
            Xh = X[:, :, h]
            Kh = np.dot(Xh[p, all_but_pq], Xh[q, all_but_pq]) - np.dot(
                Xh[all_but_pq, p], Xh[all_but_pq, q]
            )
            Gh = (
                np.linalg.norm(Xh[p, all_but_pq]) ** 2
                + np.linalg.norm(Xh[q, all_but_pq]) ** 2
                + np.linalg.norm(Xh[all_but_pq, p]) ** 2
                + np.linalg.norm(Xh[all_but_pq, q]) ** 2
            )
            xih = Xh[p, q] - Xh[q, p]

            # Build shearing matrix out of these quantities
            yk = np.arctanh((Kh - xih * dh) / (2 * (dh**2 + xih**2) + Gh))

            # Inverse of Sk on left side
            pvec = X[p, :, :].copy()
            X[p, :, :] = X[p, :, :] * np.cosh(yk) - X[q, :, :] * np.sinh(yk)
            X[q, :, :] = -pvec * np.sinh(yk) + X[q, :, :] * np.cosh(yk)

            # Sk on right side
            pvec = X[:, p, :].copy()
            X[:, p, :] = X[:, p, :] * np.cosh(yk) + X[:, q, :] * np.sinh(yk)
            X[:, q, :] = pvec * np.sinh(yk) + X[:, q, :] * np.cosh(yk)

            # Update Q_total
            pvec = Q_total[:, p].copy()
            Q_total[:, p] = Q_total[:, p] * np.cosh(yk) + Q_total[:, q] * np.sinh(yk)
            Q_total[:, q] = pvec * np.sinh(yk) + Q_total[:, q] * np.cosh(yk)

            # Defines array of off-diagonal element differences
            xi_ = -X[q, p, :] - X[p, q, :]

            # More quantities computed
            Esum = 2 * np.dot(xi_, d_)
            Dsum = np.dot(d_, d_) - np.dot(xi_, xi_)
            qt = Esum / Dsum

            th1 = np.arctan(qt)
            angle_selection = np.cos(th1) * Dsum + np.sin(th1) * Esum

            # Defines 1 of 2 possible angles
            if angle_selection > 0.0:
                theta_k = th1 / 4
            elif angle_selection < 0.0:
                theta_k = (th1 + np.pi) / 4
            else:
                raise RuntimeError("No solution found -- Jointdiag")

            # Given's rotation, this will minimize norm of off-diagonal elements only
            pvec = X[p, :, :].copy()
            X[p, :, :] = X[p, :, :] * np.cos(theta_k) - X[q, :, :] * np.sin(theta_k)
            X[q, :, :] = pvec * np.sin(theta_k) + X[q, :, :] * np.cos(theta_k)

            pvec = X[:, p, :].copy()
            X[:, p, :] = X[:, p, :] * np.cos(theta_k) - X[:, q, :] * np.sin(theta_k)
            X[:, q, :] = pvec * np.sin(theta_k) + X[:, q, :] * np.cos(theta_k)

            # Update Q_total
            pvec = Q_total[:, p].copy()
            Q_total[:, p] = Q_total[:, p] * np.cos(theta_k) - Q_total[:, q] * np.sin(
                theta_k
            )
            Q_total[:, q] = pvec * np.sin(theta_k) + Q_total[:, q] * np.cos(theta_k)

        # Error computation, check if loop needed...
        old_e = e
        e = (
            np.linalg.norm(X) ** 2.0
            - np.linalg.norm(np.diagonal(X, axis1=1, axis2=2)) ** 2.0
        )

        if verbose:
            print(f"Sweep # {k + 1}: e = {e:.3e}")

        # TODO: Strangely the error increases on the first iteration
        if old_e - e < threshold and k > 2:
            break

    return X, Q_total
