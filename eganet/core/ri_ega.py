"""
Random-Intercept EGA (riEGA).

Implements random-intercept EGA for handling wording effects.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass

from eganet.core.ega import EGA, ega_estimate


@dataclass
class RiEGAResult:
    """Result container for random-intercept EGA."""
    ega: Any
    ri_adjusted_correlation: np.ndarray
    original_correlation: np.ndarray
    ri_loadings: np.ndarray
    methods: Dict[str, Any]


def ri_ega(
    data: Union[np.ndarray, pd.DataFrame],
    n: Optional[int] = None,
    corr: str = "auto",
    na_data: str = "pairwise",
    model: str = "glasso",
    algorithm: str = "walktrap",
    verbose: bool = False,
    **kwargs
) -> RiEGAResult:
    """
    Random-Intercept Exploratory Graph Analysis.

    Adjusts for wording effects (e.g., acquiescence bias) by
    modeling a random intercept factor.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data matrix
    n : int, optional
        Sample size
    corr : str
        Correlation method
    na_data : str
        Missing data handling
    model : str
        Network estimation method
    algorithm : str
        Community detection algorithm
    verbose : bool
        Print progress

    Returns
    -------
    RiEGAResult
        Random-intercept adjusted EGA results
    """
    from eganet.correlation.auto import obtain_sample_correlations

    if isinstance(data, pd.DataFrame):
        data_arr = data.values.astype(float)
    else:
        data_arr = data.astype(float)

    corr_output = obtain_sample_correlations(
        data_arr, n, corr, na_data, verbose=verbose
    )
    original_corr = corr_output["correlation_matrix"]
    sample_size = corr_output["n"]

    n_vars = original_corr.shape[0]

    all_ones = np.ones(n_vars)
    ri_loadings = all_ones / np.sqrt(n_vars)

    ri_factor_var = np.outer(ri_loadings, ri_loadings)

    adjusted_corr = original_corr - ri_factor_var

    np.fill_diagonal(adjusted_corr, 1.0)

    from eganet.correlation.polychoric import nearest_positive_definite, is_positive_definite

    if not is_positive_definite(adjusted_corr):
        adjusted_corr = nearest_positive_definite(adjusted_corr)

    ega_result = EGA(
        adjusted_corr, n=sample_size, corr="pearson",
        model=model, algorithm=algorithm, verbose=verbose, **kwargs
    )

    methods = {
        "model": model,
        "algorithm": algorithm,
        "ri_adjustment": True,
    }

    return RiEGAResult(
        ega=ega_result,
        ri_adjusted_correlation=adjusted_corr,
        original_correlation=original_corr,
        ri_loadings=ri_loadings,
        methods=methods,
    )


def estimate_ri_loadings(
    data: np.ndarray,
    method: str = "ml"
) -> np.ndarray:
    """
    Estimate random intercept loadings.

    Parameters
    ----------
    data : np.ndarray
        Input data
    method : str
        Estimation method ("ml", "uls")

    Returns
    -------
    np.ndarray
        Estimated RI loadings
    """
    n_vars = data.shape[1]

    row_means = np.nanmean(data, axis=1, keepdims=True)

    item_corr_with_mean = np.zeros(n_vars)
    for i in range(n_vars):
        valid = ~np.isnan(data[:, i])
        if np.sum(valid) > 2:
            item_corr_with_mean[i] = np.corrcoef(
                data[valid, i], row_means[valid, 0]
            )[0, 1]

    loadings = np.abs(item_corr_with_mean)

    if np.sum(loadings) > 0:
        loadings = loadings / np.sqrt(np.sum(loadings ** 2))

    return loadings
