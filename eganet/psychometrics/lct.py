"""
Latent Class Test (LCT).

Tests whether data is better explained by a network or factor model.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any
import numpy as np


def lct(
    data: np.ndarray,
    n_factors: Optional[int] = None,
    n_boots: int = 500,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Latent Class Test.

    Tests whether data fits better to a factor model or network model.

    Parameters
    ----------
    data : np.ndarray
        Input data
    n_factors : int, optional
        Number of factors (auto-detected if None)
    n_boots : int
        Number of bootstrap samples
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        LCT results
    """
    from eganet.core.ega import EGA
    from eganet.information.tefi import tefi

    rng = np.random.default_rng(seed)
    n_samples, n_vars = data.shape

    ega_result = EGA(data)

    if n_factors is None:
        n_factors = ega_result.n_dim

    network_tefi = tefi(ega_result)["vn_entropy_fit"]

    try:
        factor_fit = _factor_model_fit(data, n_factors)
        factor_tefi = factor_fit["tefi"]
    except Exception:
        factor_tefi = np.inf

    network_tefis = []
    factor_tefis = []

    for _ in range(n_boots):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_data = data[indices]

        try:
            boot_ega = EGA(boot_data)
            boot_tefi = tefi(boot_ega)["vn_entropy_fit"]
            network_tefis.append(boot_tefi)
        except Exception:
            pass

        try:
            boot_factor = _factor_model_fit(boot_data, n_factors)
            factor_tefis.append(boot_factor["tefi"])
        except Exception:
            pass

    if len(network_tefis) > 0 and len(factor_tefis) > 0:
        network_mean = np.mean(network_tefis)
        factor_mean = np.mean(factor_tefis)

        better_model = "network" if network_tefi < factor_tefi else "factor"
        significant = abs(network_mean - factor_mean) > 2 * max(
            np.std(network_tefis), np.std(factor_tefis)
        )
    else:
        network_mean = network_tefi
        factor_mean = factor_tefi
        better_model = "network" if network_tefi < factor_tefi else "factor"
        significant = False

    return {
        "network_tefi": network_tefi,
        "factor_tefi": factor_tefi,
        "network_tefi_mean": network_mean,
        "factor_tefi_mean": factor_mean,
        "better_model": better_model,
        "significant": significant,
        "n_factors": n_factors,
        "n_dimensions": ega_result.n_dim,
    }


def _factor_model_fit(data: np.ndarray, n_factors: int) -> Dict[str, float]:
    """
    Fit factor model and compute TEFI-like measure.

    Parameters
    ----------
    data : np.ndarray
        Input data
    n_factors : int
        Number of factors

    Returns
    -------
    dict
        Factor model fit results
    """
    from eganet.correlation.auto import auto_correlate
    from eganet.information.entropy import vn_entropy

    correlation, _ = auto_correlate(data)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])

    factor_corr = np.eye(n_factors)
    implied = loadings @ factor_corr @ loadings.T
    np.fill_diagonal(implied, 1.0)

    residuals = correlation - implied

    total_vn = vn_entropy(correlation, normalized=False)
    residual_vn = vn_entropy(np.abs(residuals) + np.eye(correlation.shape[0]), normalized=False)

    tefi_like = residual_vn

    return {
        "tefi": tefi_like,
        "loadings": loadings,
        "implied": implied,
        "residuals": residuals,
        "variance_explained": np.sum(eigenvalues[:n_factors]) / np.sum(eigenvalues),
    }
