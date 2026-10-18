"""
KMO (Kaiser-Meyer-Olkin) sampling adequacy and Bartlett's test of
sphericity, implemented directly from the standard formulas.

Why hand-implemented: the usual Python package for this (`factor_analyzer`)
needs network access to install, which this development sandbox doesn't
have. Both tests are well-defined, standard statistics - implementing them
directly means they're independently testable rather than trusted as a
black box.

Both operate on a correlation matrix, so the input is (n_obs, n_vars) raw
(or standardized) data - correlation is computed internally.
"""

import numpy as np
from scipy import stats


def _correlation_matrix(X):
    return np.corrcoef(X, rowvar=False)


def kmo_test(X):
    """
    Returns (kmo_per_variable: np.ndarray, kmo_overall: float).

    KMO compares the sum of squared correlations to the sum of squared
    partial correlations (correlation between two variables with the
    effect of all others removed, derived from the inverse correlation
    matrix). Values close to 1 mean the variables share enough common
    variance for factor analysis to be meaningful; below ~0.6 is
    conventionally considered inadequate.
    """
    R = _correlation_matrix(X)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Correlation matrix is singular (likely duplicate/perfectly "
            "collinear variables in the input) - KMO is undefined."
        ) from e

    # partial correlation matrix from the inverse correlation matrix
    diag = np.sqrt(np.diag(R_inv))
    partial_corr = -R_inv / np.outer(diag, diag)
    np.fill_diagonal(partial_corr, 0)

    R_offdiag = R.copy()
    np.fill_diagonal(R_offdiag, 0)

    sum_r2 = np.sum(R_offdiag ** 2, axis=0)
    sum_partial2 = np.sum(partial_corr ** 2, axis=0)

    kmo_per_variable = sum_r2 / (sum_r2 + sum_partial2)
    kmo_overall = np.sum(R_offdiag ** 2) / (np.sum(R_offdiag ** 2) + np.sum(partial_corr ** 2))

    return kmo_per_variable, float(kmo_overall)


def bartlett_sphericity(X):
    """
    Returns (chi_square_statistic, degrees_of_freedom, p_value).

    Tests the null hypothesis that the correlation matrix is an identity
    matrix (i.e. variables are uncorrelated, so factor analysis has
    nothing to find). A significant result (p < 0.05) is a NECESSARY but
    not sufficient condition for EFA to be meaningful - it doesn't by
    itself mean the sample is large enough or the factor structure will
    be stable.
    """
    n, p = X.shape
    R = _correlation_matrix(X)
    det_R = np.linalg.det(R)

    if det_R <= 0:
        raise ValueError(
            "Correlation matrix determinant is non-positive (likely "
            "duplicate/perfectly collinear variables) - Bartlett's test "
            "is undefined."
        )

    chi_square = -(n - 1 - (2 * p + 5) / 6) * np.log(det_R)
    dof = p * (p - 1) / 2
    p_value = 1 - stats.chi2.cdf(chi_square, dof)

    return float(chi_square), float(dof), float(p_value)
