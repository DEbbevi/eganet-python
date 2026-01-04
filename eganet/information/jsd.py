"""
Jensen-Shannon Divergence.

Measures similarity between probability distributions.
"""

from __future__ import annotations
from typing import Union, Optional
import numpy as np
from scipy import stats


def jsd(
    p: np.ndarray,
    q: np.ndarray,
    base: float = np.e
) -> float:
    """
    Compute Jensen-Shannon Divergence.

    Parameters
    ----------
    p : np.ndarray
        First distribution or discrete values
    q : np.ndarray
        Second distribution or discrete values
    base : float
        Logarithm base

    Returns
    -------
    float
        Jensen-Shannon divergence (0 to 1)
    """
    if len(p) == 0 or len(q) == 0:
        return 1.0

    if np.all(np.isnan(p)) or np.all(np.isnan(q)):
        return 1.0

    p_clean = p[~np.isnan(p)]
    q_clean = q[~np.isnan(q)]

    if not _is_probability(p_clean):
        p_prob = _to_probability(p_clean)
    else:
        p_prob = p_clean

    if not _is_probability(q_clean):
        q_prob = _to_probability(q_clean)
    else:
        q_prob = q_clean

    min_len = min(len(p_prob), len(q_prob))
    p_prob = p_prob[:min_len]
    q_prob = q_prob[:min_len]

    m = (p_prob + q_prob) / 2

    jsd_val = (_kl_divergence(p_prob, m, base) + _kl_divergence(q_prob, m, base)) / 2

    return np.sqrt(max(0, min(jsd_val, 1)))


def _kl_divergence(p: np.ndarray, q: np.ndarray, base: float = np.e) -> float:
    """Compute Kullback-Leibler divergence."""
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0

    return np.sum(p[mask] * np.log(p[mask] / q[mask]) / np.log(base))


def _is_probability(x: np.ndarray) -> bool:
    """Check if array is a probability distribution."""
    if np.any(x < 0):
        return False
    if not np.isclose(np.sum(x), 1.0, atol=1e-6):
        return False
    return True


def _to_probability(x: np.ndarray) -> np.ndarray:
    """Convert discrete values to probability distribution."""
    unique, counts = np.unique(x, return_counts=True)
    return counts / len(x)


def jsd_ergodicity(
    individual_distributions: list,
    population_distribution: np.ndarray
) -> dict:
    """
    Compute ergodicity information using JSD.

    Parameters
    ----------
    individual_distributions : list
        List of individual distributions
    population_distribution : np.ndarray
        Population-level distribution

    Returns
    -------
    dict
        Ergodicity metrics
    """
    jsd_values = []

    for ind_dist in individual_distributions:
        jsd_val = jsd(ind_dist, population_distribution)
        jsd_values.append(jsd_val)

    jsd_array = np.array(jsd_values)

    return {
        "mean_jsd": np.mean(jsd_array),
        "std_jsd": np.std(jsd_array),
        "median_jsd": np.median(jsd_array),
        "ergodicity_index": 1 - np.mean(jsd_array),
        "n_individuals": len(jsd_values),
    }


def jsd_matrix(distributions: list) -> np.ndarray:
    """
    Compute pairwise JSD matrix.

    Parameters
    ----------
    distributions : list
        List of distributions

    Returns
    -------
    np.ndarray
        Pairwise JSD matrix
    """
    n = len(distributions)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            jsd_val = jsd(distributions[i], distributions[j])
            matrix[i, j] = jsd_val
            matrix[j, i] = jsd_val

    return matrix
