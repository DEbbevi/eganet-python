"""
Network fit metrics.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any
import numpy as np
from scipy import stats


def network_fit(
    network: np.ndarray,
    n: int,
    S: np.ndarray,
    ci: float = 0.95,
) -> Dict[str, float]:
    """
    Compute traditional fit metrics for networks.

    Parameters
    ----------
    network : np.ndarray
        Partial correlation network (p x p)
    n : int
        Sample size
    S : np.ndarray
        Zero-order correlation matrix (p x p)
    ci : float
        Confidence interval for RMSEA

    Returns
    -------
    dict
        Fit indices including chi-square, RMSEA, CFI, TLI, SRMR, AIC, BIC
    """
    lower_tri = np.tril_indices_from(network, k=-1)
    p = network.shape[0]

    R = partial_to_correlation(network)

    zero_parameters = p * (p - 1) / 2
    model_parameters = np.sum(network[lower_tri] != 0)

    baseline = np.eye(p)
    try:
        baseline_ml = (
            np.log(np.linalg.det(baseline)) +
            np.trace(S @ np.linalg.inv(baseline)) -
            np.log(np.linalg.det(S)) - p
        )
    except Exception:
        baseline_ml = p

    baseline_chi_square = n * baseline_ml
    baseline_tli = baseline_chi_square / zero_parameters if zero_parameters > 0 else 1

    try:
        R_inv = np.linalg.inv(R)
        loglik_ml = (
            np.log(np.linalg.det(R)) +
            np.trace(S @ R_inv) -
            np.log(np.linalg.det(S)) - p
        )
    except Exception:
        loglik_ml = 0

    chi_square = n * loglik_ml
    df = zero_parameters - model_parameters
    chi_max = max(chi_square - df, 0)
    ndf = n * df if df > 0 else 1
    rmsea_null = ndf * 0.0025

    try:
        R_inv = np.linalg.inv(R)
        loglik = -(n / 2) * (p * np.log(2 * np.pi) + np.log(np.linalg.det(R)) + np.trace(S @ R_inv))
    except Exception:
        loglik = np.nan

    rmsea = np.sqrt(chi_max / ndf) if ndf > 0 else 0
    rmsea_lower, rmsea_upper = _rmsea_ci(chi_square, df, n, ndf, ci)

    if baseline_chi_square - zero_parameters > 0:
        cfi = 1 - (chi_max / max(baseline_chi_square - zero_parameters, 1e-10))
        cfi = max(0, min(1, cfi))
    else:
        cfi = 1.0

    if baseline_tli - 1 != 0 and df > 0:
        tli = (baseline_tli - (chi_square / df)) / (baseline_tli - 1)
    else:
        tli = 1.0

    residuals = R[lower_tri] - S[lower_tri]
    srmr = np.sqrt(np.mean(residuals**2) * 2)

    p_value = 1 - stats.chi2.cdf(chi_square, df) if df > 0 else 1.0
    rmsea_p = 1 - stats.chi2.cdf(chi_max, df, rmsea_null) if df > 0 else 1.0

    ci_pct = int(ci * 100)

    return {
        "chi_square": chi_square,
        "df": df,
        "p_value": p_value,
        "rmsea": rmsea,
        f"rmsea_{ci_pct}_lower": rmsea_lower,
        f"rmsea_{ci_pct}_upper": rmsea_upper,
        "rmsea_p_value": rmsea_p,
        "cfi": cfi,
        "tli": tli,
        "srmr": srmr,
        "loglik": loglik,
        "aic": -2 * loglik + 2 * model_parameters if np.isfinite(loglik) else np.nan,
        "bic": -2 * loglik + model_parameters * np.log(n) if np.isfinite(loglik) else np.nan,
    }


def partial_to_correlation(network: np.ndarray) -> np.ndarray:
    """
    Convert partial correlation network to implied correlation matrix.

    Parameters
    ----------
    network : np.ndarray
        Partial correlation matrix

    Returns
    -------
    np.ndarray
        Implied correlation matrix
    """
    p = network.shape[0]
    P = network.copy()
    np.fill_diagonal(P, -1)

    try:
        K = -P
        K_inv = np.linalg.inv(K)
        D = np.diag(1.0 / np.sqrt(np.diag(K_inv)))
        R = D @ K_inv @ D
        np.fill_diagonal(R, 1.0)
        return R
    except Exception:
        return np.eye(p)


def _rmsea_ci(
    chi_square: float,
    df: float,
    n: int,
    ndf: float,
    ci: float
) -> tuple:
    """
    Compute RMSEA confidence intervals.

    Parameters
    ----------
    chi_square : float
        Chi-square statistic
    df : float
        Degrees of freedom
    n : int
        Sample size
    ndf : float
        n * df
    ci : float
        Confidence level

    Returns
    -------
    tuple
        (lower, upper) confidence bounds
    """
    lower_ci = 1 - (1 - ci) / 2
    upper_ci = 1 - lower_ci

    def find_lambda(lam, target):
        return stats.chi2.cdf(chi_square, df, lam) - target

    if df < 1 or find_lambda(0, lower_ci) < 0:
        rmsea_lower = 0
    else:
        try:
            from scipy.optimize import brentq
            lam = brentq(find_lambda, 0, chi_square, args=(lower_ci,))
            rmsea_lower = np.sqrt(lam / ndf)
        except Exception:
            rmsea_lower = 0

    n_rmsea = max(n, chi_square * 4)
    if df < 1 or find_lambda(n_rmsea, upper_ci) > 0 or find_lambda(0, upper_ci) < 0:
        rmsea_upper = 0
    else:
        try:
            from scipy.optimize import brentq
            lam = brentq(find_lambda, 0, n_rmsea, args=(upper_ci,))
            rmsea_upper = np.sqrt(lam / ndf)
        except Exception:
            rmsea_upper = 0

    return rmsea_lower, rmsea_upper


def fit_comparison(
    networks: Dict[str, np.ndarray],
    n: int,
    S: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compare fit indices across multiple networks.

    Parameters
    ----------
    networks : dict
        Dictionary of network matrices with names as keys
    n : int
        Sample size
    S : np.ndarray
        Correlation matrix

    Returns
    -------
    dict
        Fit indices for each network
    """
    results = {}
    for name, network in networks.items():
        results[name] = network_fit(network, n, S)
    return results
