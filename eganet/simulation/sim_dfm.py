"""
Simulate Dynamic Factor Model.

Generate time series data with factor structure.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List
import numpy as np


def sim_dfm(
    n_time: int = 100,
    n_variables: int = 10,
    n_factors: int = 2,
    n_individuals: int = 1,
    ar_coefficients: Optional[np.ndarray] = None,
    factor_loadings: Optional[np.ndarray] = None,
    noise_sd: float = 0.3,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simulate Dynamic Factor Model data.

    Parameters
    ----------
    n_time : int
        Number of time points
    n_variables : int
        Number of observed variables
    n_factors : int
        Number of latent factors
    n_individuals : int
        Number of individuals (for multi-level)
    ar_coefficients : np.ndarray, optional
        Autoregressive coefficients for factors
    factor_loadings : np.ndarray, optional
        Factor loading matrix (variables x factors)
    noise_sd : float
        Standard deviation of observation noise
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Simulated time series data
    """
    rng = np.random.default_rng(seed)

    if ar_coefficients is None:
        ar_coefficients = np.array([0.5] * n_factors)

    if factor_loadings is None:
        factor_loadings = np.zeros((n_variables, n_factors))
        vars_per_factor = n_variables // n_factors

        for f in range(n_factors):
            start = f * vars_per_factor
            end = start + vars_per_factor if f < n_factors - 1 else n_variables
            factor_loadings[start:end, f] = rng.uniform(0.4, 0.8, end - start)

    all_data = []

    for ind in range(n_individuals):
        factors = np.zeros((n_time, n_factors))
        innovations = rng.standard_normal((n_time, n_factors))

        for t in range(1, n_time):
            factors[t] = ar_coefficients * factors[t-1] + innovations[t]

        noise = rng.standard_normal((n_time, n_variables)) * noise_sd
        observed = factors @ factor_loadings.T + noise

        all_data.append(observed)

    if n_individuals == 1:
        data = all_data[0]
    else:
        data = all_data

    true_wc = np.zeros(n_variables)
    for i in range(n_variables):
        true_wc[i] = np.argmax(np.abs(factor_loadings[i])) + 1

    return {
        "data": data,
        "factors": factors if n_individuals == 1 else None,
        "factor_loadings": factor_loadings,
        "ar_coefficients": ar_coefficients,
        "true_wc": true_wc,
        "n_factors": n_factors,
    }


def sim_var(
    n_time: int = 100,
    n_variables: int = 5,
    ar_order: int = 1,
    ar_matrix: Optional[np.ndarray] = None,
    noise_cov: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simulate Vector Autoregressive (VAR) model.

    Parameters
    ----------
    n_time : int
        Number of time points
    n_variables : int
        Number of variables
    ar_order : int
        Autoregressive order
    ar_matrix : np.ndarray, optional
        AR coefficient matrix (variables x variables)
    noise_cov : np.ndarray, optional
        Innovation covariance matrix
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Simulated VAR data
    """
    rng = np.random.default_rng(seed)

    if ar_matrix is None:
        ar_matrix = np.eye(n_variables) * 0.5 + rng.uniform(-0.1, 0.1, (n_variables, n_variables))
        eigenvalues = np.linalg.eigvals(ar_matrix)
        max_abs_eig = np.max(np.abs(eigenvalues))
        if max_abs_eig >= 1:
            ar_matrix = ar_matrix / (max_abs_eig + 0.1)

    if noise_cov is None:
        noise_cov = np.eye(n_variables) * 0.5

    L = np.linalg.cholesky(noise_cov)

    data = np.zeros((n_time, n_variables))

    for t in range(1, n_time):
        innovation = L @ rng.standard_normal(n_variables)
        data[t] = ar_matrix @ data[t-1] + innovation

    return {
        "data": data,
        "ar_matrix": ar_matrix,
        "noise_cov": noise_cov,
    }
