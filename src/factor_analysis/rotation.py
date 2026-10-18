"""
Varimax (orthogonal) rotation of factor loadings.

Standard Kaiser varimax algorithm via iterative SVD, since
`factor_analyzer` isn't available offline (see stats_tests.py for why).
Rotates raw factor loadings to make each factor load strongly on a
smaller subset of variables, which is what makes factors interpretable
("this factor is mostly passing/playmaking variables") rather than each
factor spreading weight across everything.
"""

import numpy as np


def varimax(loadings, gamma=1.0, max_iter=100, tol=1e-6):
    """
    loadings: (n_vars, n_factors) unrotated factor loading matrix.
    Returns (rotated_loadings, rotation_matrix).
    """
    n_vars, n_factors = loadings.shape
    if n_factors < 2:
        return loadings.copy(), np.eye(max(n_factors, 1))

    R = np.eye(n_factors)
    d = 0.0

    for _ in range(max_iter):
        Lambda = loadings @ R
        u, s, vt = np.linalg.svd(
            loadings.T @ (
                Lambda ** 3
                - (gamma / n_vars) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))
            )
        )
        R = u @ vt
        d_new = np.sum(s)
        if d != 0 and abs(d_new - d) < tol:
            d = d_new
            break
        d = d_new

    rotated = loadings @ R
    return rotated, R
