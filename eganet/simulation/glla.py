"""
Generalized Local Linear Approximation (GLLA).

Estimates derivatives from time series using time delay embedding
and Savitzky-Golay filtering.
"""

from __future__ import annotations
from typing import Union, Optional, Literal
import math
import numpy as np

NADerivative = Literal["none", "kalman", "rowwise", "skipover"]


def glla(
    x: np.ndarray,
    n_embed: int,
    tau: int = 1,
    delta: float = 1.0,
    order: int = 2,
    na_derivative: NADerivative = "none",
) -> np.ndarray:
    """
    Generalized Local Linear Approximation.

    Estimates derivatives from time series using time delay embedding
    and a variant of Savitzky-Golay filtering.

    Parameters
    ----------
    x : np.ndarray
        Observed time series (1D array)
    n_embed : int
        Number of embedded dimensions
    tau : int
        Time delay between successive embeddings
    delta : float
        Time between successive observations
    order : int
        Maximum order of derivative to estimate
    na_derivative : str
        How to handle missing data:
        - "none": Leave NAs
        - "kalman": Use Kalman smoothing
        - "rowwise": Adaptive time intervals
        - "skipover": Skip missing and treat as continuous

    Returns
    -------
    np.ndarray
        Matrix with columns for each derivative order (0 to order)
    """
    x = np.asarray(x).flatten()

    if na_derivative == "kalman":
        x_imputed = _impute_kalman(x)
        embedding = _embed(x_imputed, n_embed, tau)
        L_matrix = _glla_setup(n_embed, tau, delta, order)
        derivative_estimates = embedding @ L_matrix

    elif na_derivative == "skipover":
        derivative_estimates = _no_correction(x, n_embed, tau, delta, order)

    elif na_derivative == "rowwise":
        derivative_estimates = _rowwise_correction(x, n_embed, tau, delta, order)

    else:
        embedding = _embed(x, n_embed, tau)
        L_matrix = _glla_setup(n_embed, tau, delta, order)
        derivative_estimates = embedding @ L_matrix

    col_names = [f"DerivOrd{i}" for i in range(order + 1)]

    return derivative_estimates


def _embed(x: np.ndarray, n_embed: int, tau: int) -> np.ndarray:
    """
    Create time delay embedding matrix.

    Parameters
    ----------
    x : np.ndarray
        Time series
    n_embed : int
        Embedding dimension
    tau : int
        Time delay

    Returns
    -------
    np.ndarray
        Embedding matrix
    """
    n = len(x)
    n_rows = n - (n_embed - 1) * tau

    if n_rows <= 0:
        raise ValueError(
            f"Time series too short for embedding. "
            f"Need at least {(n_embed - 1) * tau + 1} points."
        )

    embedding = np.zeros((n_rows, n_embed))
    for i in range(n_embed):
        start_idx = i * tau
        end_idx = start_idx + n_rows
        embedding[:, i] = x[start_idx:end_idx]

    return embedding


def _glla_setup(n_embed: int, tau: int, delta: float, order: int) -> np.ndarray:
    """
    Compute the GLLA transformation matrix.

    Parameters
    ----------
    n_embed : int
        Embedding dimension
    tau : int
        Time delay
    delta : float
        Time step
    order : int
        Maximum derivative order

    Returns
    -------
    np.ndarray
        Transformation matrix (n_embed x (order+1))
    """
    embed_seq = np.arange(1, n_embed + 1)
    embedding_value = tau * delta * embed_seq - np.mean(embed_seq)

    L = np.zeros((n_embed, order + 1))
    for d in range(order + 1):
        L[:, d] = embedding_value**d / math.factorial(d)

    return L @ np.linalg.inv(L.T @ L)


def _impute_kalman(x: np.ndarray) -> np.ndarray:
    """
    Impute missing values using Kalman smoothing.

    Parameters
    ----------
    x : np.ndarray
        Time series with potential NAs

    Returns
    -------
    np.ndarray
        Imputed time series
    """
    x = x.copy()

    if not np.any(np.isnan(x)):
        return x

    na_idx = np.isnan(x)

    if na_idx[0]:
        first_valid = np.where(~na_idx)[0]
        if len(first_valid) > 0:
            x[0] = x[first_valid[0]]

    valid_vals = x[~np.isnan(x)]
    if len(valid_vals) < 2:
        return x

    try:
        from scipy.interpolate import interp1d

        valid_idx = np.where(~np.isnan(x))[0]
        valid_vals = x[valid_idx]

        interp_func = interp1d(
            valid_idx, valid_vals,
            kind='cubic' if len(valid_idx) > 3 else 'linear',
            fill_value='extrapolate'
        )
        x[na_idx] = interp_func(np.where(na_idx)[0])

    except Exception:
        mean_val = np.nanmean(x)
        x[na_idx] = mean_val

    return x


def _no_correction(
    x: np.ndarray,
    n_embed: int,
    tau: int,
    delta: float,
    order: int
) -> np.ndarray:
    """
    Skip over missing data and treat non-missing as continuous.

    Parameters
    ----------
    x : np.ndarray
        Time series
    n_embed : int
        Embedding dimension
    tau : int
        Time delay
    delta : float
        Time step
    order : int
        Maximum derivative order

    Returns
    -------
    np.ndarray
        Derivative estimates
    """
    ts_length = len(x)
    rows = ts_length - (n_embed - 1) * tau

    derivative_matrix = np.full((rows, order + 1), np.nan)

    x_clean = x[~np.isnan(x)]

    if len(x_clean) >= n_embed:
        embedding = _embed(x_clean, n_embed, tau)
        L_matrix = _glla_setup(n_embed, tau, delta, order)
        derivatives = embedding @ L_matrix

        valid_indices = np.where(~np.isnan(x))[0]
        for i, deriv_row in enumerate(derivatives):
            if i < len(valid_indices) - (n_embed - 1) * tau:
                orig_idx = valid_indices[i]
                if orig_idx < rows:
                    derivative_matrix[orig_idx] = deriv_row

    return derivative_matrix


def _rowwise_correction(
    x: np.ndarray,
    n_embed: int,
    tau: int,
    delta: float,
    order: int
) -> np.ndarray:
    """
    Adaptive time intervals with respect to missing data.

    Parameters
    ----------
    x : np.ndarray
        Time series
    n_embed : int
        Embedding dimension
    tau : int
        Time delay
    delta : float
        Time step
    order : int
        Maximum derivative order

    Returns
    -------
    np.ndarray
        Derivative estimates
    """
    full_length = len(x)
    row_seq = np.arange(full_length)

    row_index = _embed(row_seq, n_embed, tau)

    available_idx = ~np.isnan(x)
    x_available = x[available_idx]
    time_available = np.where(available_idx)[0]

    if len(x_available) < n_embed:
        return np.full((len(row_index), order + 1), np.nan)

    embedding = _embed(x_available, n_embed, tau)
    time_embed = _embed(time_available.astype(float), n_embed, tau)

    derivative_row = []
    for i in range(len(time_embed)):
        match_idx = np.where(row_index[:, 0] == time_embed[i, 0])[0]
        if len(match_idx) > 0:
            derivative_row.append(match_idx[0])
        else:
            derivative_row.append(-1)

    derivative_matrix = np.full((len(row_index), order + 1), np.nan)

    intervals = np.sum(time_embed, axis=1) / 2

    for i in range(len(embedding)):
        if derivative_row[i] < 0:
            continue

        center = time_embed[i] - intervals[i]

        L = np.ones((n_embed, 1))
        if order >= 1:
            L = np.column_stack([L, center])
        if order >= 2:
            L = np.column_stack([L, center**2])

        try:
            transform = L @ np.linalg.inv(L.T @ L)
            derivative_matrix[derivative_row[i]] = embedding[i] @ transform
        except Exception:
            pass

    return derivative_matrix


def derivative_estimates(
    data: np.ndarray,
    n_embed: int = 5,
    tau: int = 1,
    delta: float = 1.0,
    order: int = 2,
) -> np.ndarray:
    """
    Compute derivative estimates for multivariate time series.

    Parameters
    ----------
    data : np.ndarray
        Multivariate time series (n_time x n_vars)
    n_embed : int
        Embedding dimension
    tau : int
        Time delay
    delta : float
        Time step
    order : int
        Maximum derivative order

    Returns
    -------
    np.ndarray
        Derivative estimates for each variable
    """
    data = np.atleast_2d(data)
    if data.shape[0] < data.shape[1]:
        data = data.T

    n_time, n_vars = data.shape
    n_rows = n_time - (n_embed - 1) * tau

    all_derivatives = []

    for var in range(n_vars):
        deriv = glla(data[:, var], n_embed, tau, delta, order)
        all_derivatives.append(deriv)

    return np.stack(all_derivatives, axis=2)
