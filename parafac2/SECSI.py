from itertools import combinations

import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao, mode_dot

from .jointdiag import jointdiag


def SECSI(X, d: int, maxIter: int = 50, tolerance=1e-12, verbose=True):
    """
    Computes Semi-Algebraic CP factorization using a joint diagonalization algorithm:

    Args:
    X: 3-mode tensor to be factorized
    d: estimated rank of factorization
    maxIter: number of jointdiag iterations to compute per estimate
    """

    R = X.ndim  # Number of modes

    # Compute truncated higher order SVD, cut off to d(rank) elements
    core_trunc, factors_trunc = tl.decomposition.tucker(X, rank=[d] * len(X.shape))

    # Initialize dataframe for estimate matrices
    f_estimates = []
    norm_est = []

    # Loop finds all valid Simultaneuous Matrix Diagonalizations(SMDs)
    for k_mode, l_mode in combinations(range(R), 2):
        # Find all combinations of modes (k,l) longer than or equal to the rank d
        if X.shape[k_mode] < d or X.shape[l_mode] < d:
            continue

        # Find 3rd mode
        mode_not_kl = list(set(range(R)) - {k_mode, l_mode})[0]

        # Compute n-mode product between core and factor matrices for 3rd mode
        Skl = mode_dot(core_trunc, factors_trunc[mode_not_kl], mode_not_kl)

        # Rearranges tensor so that k,l are first 2 modes
        SMD = Skl.transpose(k_mode, l_mode, mode_not_kl)

        # Compute 2 norm condition number for each matrix slice, save values
        conds = np.linalg.cond(SMD.T)

        ###
        # Using computed SMDs, we now generate factor matrices
        # through joint diagonalization

        # Save matrix slice with minimal norm
        optimal_slice = SMD[:, :, np.argmin(conds)]  # Pivot Slice

        # Sets up left and right hand side SMDs using pivot slice
        # Solves matrix equation optimal * X = n-th slice --> n-th slice / optimal
        # Solves lhs version, optimal * X = n-th slice ^T --> optimal / n-th slice
        SMD_rhs = np.linalg.solve(optimal_slice.T, SMD.T).T
        SMD_lhs = np.linalg.solve(optimal_slice, np.moveaxis(SMD, 2, 0)).T

        for SMD_sel, first_mode, second_mode in zip(
            [SMD_rhs, SMD_lhs], (k_mode, l_mode), (l_mode, k_mode), strict=False
        ):
            # Compute joint diagonalization of all matrix slices in SMD
            Diags, Transform = jointdiag(
                SMD_sel,
                MaxIter=maxIter,
                threshold=tolerance,
                verbose=verbose,
            )

            cp_tensor = tl.cp_tensor.CPTensor(
                (None, [np.zeros_like(f) for f in factors_trunc])
            )

            # Now compute two estimates of all three factor matrices...
            # First estimate based on factor * transform matrix
            cp_tensor.factors[first_mode] = factors_trunc[first_mode] @ Transform

            # Picks out diagonal of n-th slice of SMD, saves n-th row of krp matrix
            cp_tensor.factors[mode_not_kl] = np.diagonal(Diags)

            # Khatri rao product of other two estimates
            if first_mode < mode_not_kl:
                krp = khatri_rao(
                    (cp_tensor.factors[first_mode], cp_tensor.factors[mode_not_kl])
                )
            else:
                krp = khatri_rao(
                    (cp_tensor.factors[mode_not_kl], cp_tensor.factors[first_mode])
                )

            # Estimates final factor matrix by solving least squares matrix equation
            # using least squares solution to X * krp = unfolding
            cp_tensor.factors[second_mode], resid, _, _ = np.linalg.lstsq(
                krp, tl.unfold(X, second_mode).T, rcond=None
            )
            cp_tensor.factors[second_mode] = cp_tensor.factors[second_mode].T

            f_estimates.append(cp_tensor)
            norm_est.append(resid.sum())

    # Stops if previous loop found nothing
    if len(norm_est) == 0:
        raise Warning("No SMDs found, too many rank deficiencies")

    # TODO: Sort f_ests based on error
    return norm_est, f_estimates
