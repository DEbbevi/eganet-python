"""
Cosine similarity computation.

Provides cosine similarity as an alternative similarity measure
for network construction.
"""

from __future__ import annotations
import numpy as np
from typing import Union
import pandas as pd


def cosine_similarity(
    data: Union[np.ndarray, pd.DataFrame],
    center: bool = True
) -> np.ndarray:
    """
    Compute cosine similarity matrix.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data matrix (samples x variables)
    center : bool
        Whether to center data (mean subtract) before computing

    Returns
    -------
    np.ndarray
        Cosine similarity matrix (variables x variables)
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    data = data.astype(float)

    mask = ~np.isnan(data)
    data = np.where(mask, data, 0)

    if center:
        col_means = np.nanmean(data, axis=0, keepdims=True)
        data = data - col_means

    norms = np.linalg.norm(data, axis=0, keepdims=True)
    norms[norms == 0] = 1

    normalized = data / norms

    similarity = normalized.T @ normalized

    np.fill_diagonal(similarity, 1.0)

    return similarity


def pairwise_cosine(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Parameters
    ----------
    x : np.ndarray
        First vector
    y : np.ndarray
        Second vector

    Returns
    -------
    float
        Cosine similarity
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[mask]
    y_valid = y[mask]

    if len(x_valid) == 0:
        return np.nan

    norm_x = np.linalg.norm(x_valid)
    norm_y = np.linalg.norm(y_valid)

    if norm_x == 0 or norm_y == 0:
        return 0.0

    return np.dot(x_valid, y_valid) / (norm_x * norm_y)
