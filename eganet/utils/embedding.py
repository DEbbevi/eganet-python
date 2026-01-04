"""
Time series embedding for dynamic EGA.
"""

from __future__ import annotations
from typing import Union, Optional
import numpy as np
import pandas as pd


def embed(
    data: Union[np.ndarray, pd.DataFrame],
    n_embed: int = 5,
    tau: int = 1,
) -> np.ndarray:
    """
    Time-delay embedding for multivariate time series.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Time series data (time x variables)
    n_embed : int
        Embedding dimension
    tau : int
        Time delay

    Returns
    -------
    np.ndarray
        Embedded data matrix
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    n_time, n_vars = data.shape

    n_embedded = n_time - (n_embed - 1) * tau
    if n_embedded <= 0:
        raise ValueError(f"Not enough time points for embedding. Need at least {(n_embed - 1) * tau + 1}")

    embedded = np.zeros((n_embedded, n_vars * n_embed))

    for i in range(n_embed):
        start_idx = i * tau
        end_idx = start_idx + n_embedded
        col_start = i * n_vars
        col_end = col_start + n_vars
        embedded[:, col_start:col_end] = data[start_idx:end_idx, :]

    return embedded


def optimal_embed_dimension(
    data: np.ndarray,
    max_embed: int = 10,
    method: str = "tefi"
) -> int:
    """
    Find optimal embedding dimension using TEFI.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    max_embed : int
        Maximum embedding dimension to try
    method : str
        Optimization method ("tefi")

    Returns
    -------
    int
        Optimal embedding dimension
    """
    from eganet.core.ega import ega_estimate
    from eganet.information.tefi import tefi

    best_tefi = np.inf
    best_embed = 1

    for n_embed in range(1, max_embed + 1):
        try:
            embedded = embed(data, n_embed=n_embed)

            if embedded.shape[0] < embedded.shape[1]:
                break

            result = ega_estimate(embedded)
            tefi_val = tefi(result)["vn_entropy_fit"]

            if tefi_val < best_tefi:
                best_tefi = tefi_val
                best_embed = n_embed

        except Exception:
            continue

    return best_embed


def kalman_smooth(
    data: np.ndarray,
    missing_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Simple Kalman smoothing for missing data imputation.

    Parameters
    ----------
    data : np.ndarray
        Time series with missing values
    missing_mask : np.ndarray, optional
        Boolean mask of missing values

    Returns
    -------
    np.ndarray
        Smoothed data with imputed values
    """
    if missing_mask is None:
        missing_mask = np.isnan(data)

    result = data.copy()

    for j in range(data.shape[1]):
        col = result[:, j]
        missing = missing_mask[:, j]

        if not np.any(missing):
            continue

        valid_indices = np.where(~missing)[0]
        missing_indices = np.where(missing)[0]

        if len(valid_indices) < 2:
            col_mean = np.nanmean(col)
            col[missing] = col_mean
            continue

        for idx in missing_indices:
            before = valid_indices[valid_indices < idx]
            after = valid_indices[valid_indices > idx]

            if len(before) > 0 and len(after) > 0:
                before_idx = before[-1]
                after_idx = after[0]
                weight = (idx - before_idx) / (after_idx - before_idx)
                col[idx] = col[before_idx] * (1 - weight) + col[after_idx] * weight
            elif len(before) > 0:
                col[idx] = col[before[-1]]
            elif len(after) > 0:
                col[idx] = col[after[0]]
            else:
                col[idx] = np.nanmean(col)

        result[:, j] = col

    return result
