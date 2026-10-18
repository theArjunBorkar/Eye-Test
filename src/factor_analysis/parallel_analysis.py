"""
Parallel analysis (Horn's method) for choosing how many factors to retain.

Simulates many random datasets of the same shape as the real data, takes
the eigenvalues of their correlation matrices, and compares each real
eigenvalue against the corresponding percentile (default 95th) of the
simulated eigenvalues at that rank. A factor is retained only if its real
eigenvalue exceeds what pure random noise would produce - this is
considered more reliable than the simpler Kaiser criterion
(eigenvalue > 1), which tends to over-retain factors.
"""

import numpy as np


def parallel_analysis(n_obs, n_vars, n_iter=200, percentile=95, random_state=42):
    """
    Returns an array of length n_vars: the `percentile`-th percentile of
    simulated eigenvalues at each rank, from `n_iter` random datasets of
    shape (n_obs, n_vars).
    """
    rng = np.random.default_rng(random_state)
    simulated_eigenvalues = np.zeros((n_iter, n_vars))

    for i in range(n_iter):
        random_data = rng.standard_normal((n_obs, n_vars))
        corr = np.corrcoef(random_data, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(corr)[::-1]  # descending
        simulated_eigenvalues[i] = eigenvalues

    return np.percentile(simulated_eigenvalues, percentile, axis=0)


def suggest_n_factors(real_eigenvalues, simulated_threshold_eigenvalues):
    """
    Number of factors where the real eigenvalue exceeds the simulated
    threshold, stopping at the first factor that doesn't (standard
    parallel analysis stopping rule - not "count all that ever exceed").
    """
    n = 0
    for real, threshold in zip(real_eigenvalues, simulated_threshold_eigenvalues):
        if real > threshold:
            n += 1
        else:
            break
    return n
