"""
Confirmatory Factor Analysis comparison for EGA.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from eganet.utils.helpers import EGAResult


def cfa(
    ega_result: "EGAResult",
    data: Optional[np.ndarray] = None,
    estimator: str = "ml",
) -> Dict[str, Any]:
    """
    Perform CFA based on EGA structure.

    Parameters
    ----------
    ega_result : EGAResult
        EGA result with structure
    data : np.ndarray, optional
        Original data (uses correlation if not provided)
    estimator : str
        Estimation method

    Returns
    -------
    dict
        CFA results including fit indices
    """
    wc = ega_result.wc
    correlation = ega_result.correlation
    n = ega_result.n

    n_vars = len(wc)
    valid_wc = wc[~np.isnan(wc)]
    communities = np.unique(valid_wc)
    n_factors = len(communities)

    loading_matrix = np.zeros((n_vars, n_factors))
    for i, comm in enumerate(communities):
        mask = wc == comm
        loading_matrix[mask, i] = 1.0

    n_params = np.sum(loading_matrix > 0) + n_factors * (n_factors - 1) / 2

    try:
        factor_corr = np.eye(n_factors)
        for i, comm_i in enumerate(communities):
            for j, comm_j in enumerate(communities):
                if i < j:
                    mask_i = wc == comm_i
                    mask_j = wc == comm_j
                    submatrix = correlation[np.ix_(mask_i, mask_j)]
                    factor_corr[i, j] = np.mean(np.abs(submatrix))
                    factor_corr[j, i] = factor_corr[i, j]

        implied = loading_matrix @ factor_corr @ loading_matrix.T

        residuals = correlation - implied
        np.fill_diagonal(residuals, 0)

        chi_sq = n * np.sum(residuals ** 2)
        df = n_vars * (n_vars - 1) / 2 - n_params

        if df > 0:
            rmsea = np.sqrt(max(0, (chi_sq / df - 1) / (n - 1)))
        else:
            rmsea = 0

        null_chi_sq = n * np.sum((correlation - np.eye(n_vars)) ** 2) / 2
        if null_chi_sq > chi_sq:
            cfi = 1 - (chi_sq - df) / (null_chi_sq - n_vars * (n_vars - 1) / 2)
            cfi = max(0, min(1, cfi))
        else:
            cfi = 1.0

        srmr = np.sqrt(np.mean(residuals ** 2) * 2)

        fit_indices = {
            "chi_square": chi_sq,
            "df": df,
            "p_value": 1 - _chi2_cdf(chi_sq, df) if df > 0 else 1.0,
            "rmsea": rmsea,
            "cfi": cfi,
            "srmr": srmr,
            "n_factors": n_factors,
            "n_params": n_params,
        }

        fit_acceptable = rmsea < 0.08 and cfi > 0.90 and srmr < 0.08

    except Exception as e:
        fit_indices = {
            "error": str(e),
            "n_factors": n_factors,
        }
        fit_acceptable = False

    return {
        "fit_indices": fit_indices,
        "fit_acceptable": fit_acceptable,
        "loading_matrix": loading_matrix,
        "factor_correlation": factor_corr if "factor_corr" in dir() else None,
    }


def _chi2_cdf(x: float, df: float) -> float:
    """Compute chi-squared CDF."""
    from scipy import stats
    return stats.chi2.cdf(x, df)


def compare_structures(
    ega_result: "EGAResult",
    alternative_structures: List[np.ndarray],
    data: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compare EGA structure with alternative structures using CFA fit.

    Parameters
    ----------
    ega_result : EGAResult
        EGA result
    alternative_structures : list
        List of alternative community membership vectors
    data : np.ndarray, optional
        Original data

    Returns
    -------
    pd.DataFrame
        Comparison of fit indices
    """
    from eganet.utils.helpers import EGAResult

    results = []

    ega_cfa = cfa(ega_result, data)
    results.append({
        "structure": "EGA",
        **ega_cfa["fit_indices"],
        "acceptable": ega_cfa["fit_acceptable"],
    })

    for i, alt_wc in enumerate(alternative_structures):
        alt_result = EGAResult(
            network=ega_result.network,
            wc=alt_wc,
            n_dim=len(np.unique(alt_wc[~np.isnan(alt_wc)])),
            correlation=ega_result.correlation,
            n=ega_result.n,
            dim_variables=ega_result.dim_variables,
        )

        alt_cfa = cfa(alt_result, data)
        results.append({
            "structure": f"Alternative_{i+1}",
            **alt_cfa["fit_indices"],
            "acceptable": alt_cfa["fit_acceptable"],
        })

    return pd.DataFrame(results)
