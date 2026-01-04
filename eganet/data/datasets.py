"""
Sample datasets for EGAnet.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd


def load_wmt2(as_dataframe: bool = True):
    """
    Load simulated WMT2 dataset.

    This is a simulated dataset similar to the Woodcock-Muñoz Test
    of Cognitive Abilities used in the original EGAnet R package.

    Parameters
    ----------
    as_dataframe : bool
        Return as DataFrame (True) or dict (False)

    Returns
    -------
    pd.DataFrame or dict
        Dataset with simulated cognitive ability data
    """
    np.random.seed(42)

    n_samples = 500
    n_items_per_factor = 6
    n_factors = 3

    loadings = np.array([
        [0.7, 0.1, 0.0],
        [0.8, 0.0, 0.1],
        [0.6, 0.1, 0.0],
        [0.7, 0.0, 0.1],
        [0.8, 0.1, 0.0],
        [0.7, 0.0, 0.0],
        [0.1, 0.7, 0.0],
        [0.0, 0.8, 0.1],
        [0.1, 0.6, 0.0],
        [0.0, 0.7, 0.1],
        [0.1, 0.8, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, 0.1, 0.7],
        [0.1, 0.0, 0.8],
        [0.0, 0.1, 0.6],
        [0.1, 0.0, 0.7],
        [0.0, 0.1, 0.8],
        [0.0, 0.0, 0.7],
    ])

    factor_corr = np.array([
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ])

    L_factor = np.linalg.cholesky(factor_corr)
    factors = np.random.randn(n_samples, n_factors) @ L_factor.T

    unique_var = 1 - np.sum(loadings ** 2, axis=1)
    unique_var = np.maximum(unique_var, 0.1)
    noise = np.random.randn(n_samples, 18) * np.sqrt(unique_var)

    data = factors @ loadings.T + noise

    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data = data * 10 + 50
    data = np.round(data).astype(int)
    data = np.clip(data, 1, 100)

    var_names = [f"Item_{i+1}" for i in range(18)]

    if as_dataframe:
        df = pd.DataFrame(data, columns=var_names)
        df["ID"] = range(1, n_samples + 1)
        df["Age"] = np.random.randint(18, 65, n_samples)
        df["Gender"] = np.random.choice(["M", "F"], n_samples)

        cols = ["ID", "Age", "Gender"] + var_names
        return df[cols]

    return {
        "data": data,
        "var_names": var_names,
        "n_samples": n_samples,
        "n_items": 18,
        "true_structure": np.repeat([1, 2, 3], 6),
    }


def load_depression(as_dataframe: bool = True):
    """
    Load simulated depression symptom dataset.

    Parameters
    ----------
    as_dataframe : bool
        Return as DataFrame

    Returns
    -------
    pd.DataFrame or dict
        Depression symptom data
    """
    np.random.seed(123)

    n_samples = 300
    n_items = 9

    loadings = np.array([
        [0.8, 0.1],
        [0.7, 0.2],
        [0.6, 0.1],
        [0.7, 0.1],
        [0.8, 0.0],
        [0.1, 0.8],
        [0.2, 0.7],
        [0.1, 0.6],
        [0.0, 0.7],
    ])

    factor_corr = np.array([[1.0, 0.4], [0.4, 1.0]])
    L = np.linalg.cholesky(factor_corr)
    factors = np.random.randn(n_samples, 2) @ L.T

    unique_var = 1 - np.sum(loadings ** 2, axis=1)
    unique_var = np.maximum(unique_var, 0.1)
    noise = np.random.randn(n_samples, n_items) * np.sqrt(unique_var)

    data = factors @ loadings.T + noise

    data = (data - data.min()) / (data.max() - data.min())
    data = np.round(data * 4).astype(int)
    data = np.clip(data, 0, 4)

    var_names = [
        "depressed_mood", "loss_interest", "weight_change",
        "sleep_problems", "fatigue", "guilt", "concentration",
        "restlessness", "suicidal_thoughts"
    ]

    if as_dataframe:
        return pd.DataFrame(data, columns=var_names)

    return {
        "data": data,
        "var_names": var_names,
        "true_structure": np.array([1, 1, 1, 1, 1, 2, 2, 2, 2]),
    }


def load_optimism(as_dataframe: bool = True):
    """
    Load simulated optimism scale dataset.

    Parameters
    ----------
    as_dataframe : bool
        Return as DataFrame

    Returns
    -------
    pd.DataFrame or dict
        Optimism scale data
    """
    np.random.seed(456)

    n_samples = 400
    n_items = 10

    loadings = np.zeros((n_items, 2))
    loadings[:5, 0] = np.random.uniform(0.6, 0.8, 5)
    loadings[5:, 1] = np.random.uniform(0.6, 0.8, 5)

    factor_corr = np.array([[1.0, -0.5], [-0.5, 1.0]])
    L = np.linalg.cholesky(factor_corr)
    factors = np.random.randn(n_samples, 2) @ L.T

    unique_var = 1 - np.sum(loadings ** 2, axis=1)
    unique_var = np.maximum(unique_var, 0.2)
    noise = np.random.randn(n_samples, n_items) * np.sqrt(unique_var)

    data = factors @ loadings.T + noise

    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data = np.round(data * 0.8 + 3).astype(int)
    data = np.clip(data, 1, 5)

    var_names = [f"Opt_{i+1}" for i in range(5)] + [f"Pes_{i+1}" for i in range(5)]

    if as_dataframe:
        return pd.DataFrame(data, columns=var_names)

    return {
        "data": data,
        "var_names": var_names,
        "true_structure": np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
    }


def load_simulation_data(
    n_factors: int = 3,
    n_items_per_factor: int = 5,
    n_samples: int = 500,
    loading_range: tuple = (0.5, 0.8),
    factor_correlation: float = 0.3,
    seed: int = None,
) -> Dict[str, Any]:
    """
    Generate custom simulation data.

    Parameters
    ----------
    n_factors : int
        Number of factors
    n_items_per_factor : int
        Items per factor
    n_samples : int
        Sample size
    loading_range : tuple
        Range for factor loadings
    factor_correlation : float
        Correlation between factors
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Simulated data and parameters
    """
    if seed is not None:
        np.random.seed(seed)

    n_items = n_factors * n_items_per_factor

    loadings = np.zeros((n_items, n_factors))
    for f in range(n_factors):
        start = f * n_items_per_factor
        end = start + n_items_per_factor
        loadings[start:end, f] = np.random.uniform(*loading_range, n_items_per_factor)

    factor_corr = np.eye(n_factors) * (1 - factor_correlation) + factor_correlation
    np.fill_diagonal(factor_corr, 1.0)

    L = np.linalg.cholesky(factor_corr)
    factors = np.random.randn(n_samples, n_factors) @ L.T

    unique_var = 1 - np.sum(loadings ** 2, axis=1)
    unique_var = np.maximum(unique_var, 0.1)
    noise = np.random.randn(n_samples, n_items) * np.sqrt(unique_var)

    data = factors @ loadings.T + noise

    true_structure = np.repeat(np.arange(1, n_factors + 1), n_items_per_factor)

    return {
        "data": data,
        "loadings": loadings,
        "factor_correlation": factor_corr,
        "true_structure": true_structure,
        "n_samples": n_samples,
        "n_items": n_items,
        "n_factors": n_factors,
    }
