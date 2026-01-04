"""
Polychoric correlation computation.

Implements polychoric correlations for ordinal data using:
- Beasley-Springer-Moro algorithm for inverse normal CDF
- Drezner-Wesolowsky approximation for bivariate normal CDF
- Brent's method for optimization

This is a pure Python port of the EGAnet C implementation.
"""

from __future__ import annotations
from typing import Union, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.optimize import brentq
import warnings


CONST_A = np.array([
    -39.69683028665376, 220.9460984245205, -275.928510446969,
    138.357751867269, -30.66479806614716, 2.506628277459239
])
CONST_B = np.array([
    -54.47609879822406, 161.5858368580409, -155.6989798598866,
    66.80131188771972, -13.28068155288572
])
CONST_C = np.array([
    -0.007784894002430293, -0.3223964580411365, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783
])
CONST_D = np.array([
    0.007784695709041462, 0.3224671290700398, 2.445134137142996,
    3.754408661907416
])

DOUBLE_X = np.array([0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992])
DOUBLE_W = np.array([0.018854042, 0.038088059, 0.0452707394, 0.038088059, 0.018854042])


def bsm_inverse_cdf(probability: float) -> float:
    """
    Inverse normal CDF using Beasley-Springer-Moro algorithm.

    Parameters
    ----------
    probability : float
        Probability value between 0 and 1

    Returns
    -------
    float
        Inverse normal CDF value
    """
    if probability <= 0:
        return -np.inf
    if probability >= 1:
        return np.inf

    not_lower_tail = probability >= 0.02425

    if not_lower_tail and probability <= 0.97575:
        q = probability - 0.5
        r = q * q
        x = (((((CONST_A[0] * r + CONST_A[1]) * r + CONST_A[2]) * r +
               CONST_A[3]) * r + CONST_A[4]) * r + CONST_A[5]) * q
        x /= ((((CONST_B[0] * r + CONST_B[1]) * r + CONST_B[2]) * r +
               CONST_B[3]) * r + CONST_B[4]) * r + 1
    else:
        if not_lower_tail:
            q = np.sqrt(-2 * np.log(1 - probability))
        else:
            q = np.sqrt(-2 * np.log(probability))

        x = (((((CONST_C[0] * q + CONST_C[1]) * q + CONST_C[2]) * q +
               CONST_C[3]) * q + CONST_C[4]) * q + CONST_C[5])
        x /= (((CONST_D[0] * q + CONST_D[1]) * q + CONST_D[2]) * q +
              CONST_D[3]) * q + 1

        if not_lower_tail:
            x = -x

    return x


def drezner_bivariate_normal(
    lower_x: float,
    lower_y: float,
    rho: float
) -> float:
    """
    Bivariate normal CDF using Drezner-Wesolowsky approximation.

    Parameters
    ----------
    lower_x : float
        Lower bound for X
    lower_y : float
        Lower bound for Y
    rho : float
        Correlation coefficient

    Returns
    -------
    float
        Bivariate normal CDF value P(X < lower_x, Y < lower_y)
    """
    if np.abs(rho) < 1e-10:
        return stats.norm.cdf(lower_x) * stats.norm.cdf(lower_y)

    if np.abs(rho) > 0.9999:
        if rho > 0:
            return stats.norm.cdf(min(lower_x, lower_y))
        else:
            return max(0, stats.norm.cdf(lower_x) - stats.norm.cdf(-lower_y))

    hs = (lower_x * lower_x + lower_y * lower_y) / 2
    asr = np.arcsin(rho)

    result = 0.0
    for i in range(5):
        for sign in [-1, 1]:
            sn = np.sin(asr * (sign * DOUBLE_X[i] + 1) / 2)
            result += DOUBLE_W[i] * np.exp((sn * lower_x * lower_y - hs) / (1 - sn * sn))

    result = result * asr / (4 * np.pi)
    result += stats.norm.cdf(-lower_x) * stats.norm.cdf(-lower_y)

    return max(0, min(1, result))


def compute_thresholds(
    marginal_counts: np.ndarray,
    n: int
) -> np.ndarray:
    """
    Compute thresholds from marginal frequencies.

    Parameters
    ----------
    marginal_counts : np.ndarray
        Marginal frequency counts
    n : int
        Total sample size

    Returns
    -------
    np.ndarray
        Threshold values (inverse normal of cumulative proportions)
    """
    cum_prop = np.cumsum(marginal_counts) / n

    thresholds = np.full(len(cum_prop) + 1, np.nan)
    thresholds[0] = -np.inf
    thresholds[-1] = np.inf

    for i, p in enumerate(cum_prop[:-1]):
        if p <= 0:
            thresholds[i + 1] = -np.inf
        elif p >= 1:
            thresholds[i + 1] = np.inf
        else:
            thresholds[i + 1] = bsm_inverse_cdf(p)

    return thresholds


def expected_frequency(
    thresh_x: np.ndarray,
    thresh_y: np.ndarray,
    rho: float
) -> np.ndarray:
    """
    Compute expected frequencies given thresholds and correlation.

    Parameters
    ----------
    thresh_x : np.ndarray
        Thresholds for X variable
    thresh_y : np.ndarray
        Thresholds for Y variable
    rho : float
        Polychoric correlation

    Returns
    -------
    np.ndarray
        Expected frequency matrix
    """
    nx = len(thresh_x) - 1
    ny = len(thresh_y) - 1
    expected = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            p11 = drezner_bivariate_normal(thresh_x[i + 1], thresh_y[j + 1], rho)
            p10 = drezner_bivariate_normal(thresh_x[i + 1], thresh_y[j], rho)
            p01 = drezner_bivariate_normal(thresh_x[i], thresh_y[j + 1], rho)
            p00 = drezner_bivariate_normal(thresh_x[i], thresh_y[j], rho)

            expected[i, j] = p11 - p10 - p01 + p00

    return expected


def polychoric_correlation_pair(
    x: np.ndarray,
    y: np.ndarray,
    empty_correction: float = 0.5
) -> Tuple[float, bool]:
    """
    Compute polychoric correlation between two ordinal variables.

    Parameters
    ----------
    x : np.ndarray
        First ordinal variable
    y : np.ndarray
        Second ordinal variable
    empty_correction : float
        Correction to add to empty cells

    Returns
    -------
    tuple
        (polychoric correlation, convergence flag)
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[mask].astype(int)
    y_valid = y[mask].astype(int)

    if len(x_valid) < 3:
        return np.nan, False

    x_min, x_max = x_valid.min(), x_valid.max()
    y_min, y_max = y_valid.min(), y_valid.max()

    x_shifted = x_valid - x_min
    y_shifted = y_valid - y_min

    nx = x_max - x_min + 1
    ny = y_max - y_min + 1

    if nx < 2 or ny < 2:
        return np.nan, False

    joint_freq = np.zeros((nx, ny), dtype=float)
    for xi, yi in zip(x_shifted, y_shifted):
        joint_freq[xi, yi] += 1

    if empty_correction > 0:
        joint_freq[joint_freq == 0] = empty_correction

    n = joint_freq.sum()
    marginal_x = joint_freq.sum(axis=1)
    marginal_y = joint_freq.sum(axis=0)

    thresh_x = compute_thresholds(marginal_x, n)
    thresh_y = compute_thresholds(marginal_y, n)

    observed_prop = joint_freq / n

    def objective(rho):
        expected = expected_frequency(thresh_x, thresh_y, rho)
        expected = np.maximum(expected, 1e-10)
        log_like = np.sum(observed_prop * np.log(expected))
        return -log_like

    try:
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(
            objective,
            bounds=(-0.999, 0.999),
            method='bounded',
            options={'xatol': 1e-6}
        )
        return result.x, result.success
    except Exception:
        return np.corrcoef(x_valid, y_valid)[0, 1], False


def polychoric_matrix(
    data: np.ndarray,
    ordinal_categories: int = 7,
    empty_correction: float = 0.5,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute polychoric correlation matrix.

    Parameters
    ----------
    data : np.ndarray
        Data matrix with ordinal variables
    ordinal_categories : int
        Maximum categories to consider ordinal
    empty_correction : float
        Correction for empty cells
    verbose : bool
        Print progress

    Returns
    -------
    np.ndarray
        Polychoric correlation matrix
    """
    n_vars = data.shape[1]
    corr_matrix = np.eye(n_vars)

    total_pairs = n_vars * (n_vars - 1) // 2
    pair_count = 0

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            pair_count += 1
            if verbose and pair_count % 100 == 0:
                print(f"Processing pair {pair_count}/{total_pairs}")

            rho, converged = polychoric_correlation_pair(
                data[:, i], data[:, j], empty_correction
            )

            if np.isnan(rho):
                rho = 0.0

            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    if np.any(eigenvalues < -1e-10):
        try:
            from scipy.linalg import sqrtm
            corr_matrix = nearest_positive_definite(corr_matrix)
        except Exception:
            pass

    return corr_matrix


def tetrachoric_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute tetrachoric correlation for binary variables.

    Parameters
    ----------
    x : np.ndarray
        First binary variable
    y : np.ndarray
        Second binary variable

    Returns
    -------
    float
        Tetrachoric correlation
    """
    return polychoric_correlation_pair(x, y)[0]


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """
    Find nearest positive definite matrix.

    Uses Higham's algorithm.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix

    Returns
    -------
    np.ndarray
        Nearest positive definite matrix
    """
    B = (matrix + matrix.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(matrix))
    I = np.eye(matrix.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
        if k > 100:
            break

    return A3


def is_positive_definite(matrix: np.ndarray) -> bool:
    """Check if matrix is positive definite."""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
