"""
Stability assessment for EGA.

Provides dimension and item stability measures from bootstrap results.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List, TYPE_CHECKING
import numpy as np
import pandas as pd
from collections import Counter

if TYPE_CHECKING:
    from eganet.utils.helpers import BootEGAResult, EGAResult


def dimension_stability(
    boot_result: "BootEGAResult",
) -> Dict[str, Any]:
    """
    Compute dimension stability from bootstrap results.

    Parameters
    ----------
    boot_result : BootEGAResult
        Bootstrap EGA results

    Returns
    -------
    dict
        Dimension stability metrics
    """
    n_dims = [r.n_dim for r in boot_result.boot_results]

    counts = Counter(n_dims)
    total = len(n_dims)

    frequency = {dim: count / total for dim, count in counts.items()}

    modal_dims = counts.most_common(1)[0][0]
    modal_frequency = counts.most_common(1)[0][1] / total

    mean_dims = np.mean(n_dims)
    std_dims = np.std(n_dims)

    return {
        "frequency": frequency,
        "modal_dimensions": modal_dims,
        "modal_frequency": modal_frequency,
        "mean_dimensions": mean_dims,
        "std_dimensions": std_dims,
        "min_dimensions": min(n_dims),
        "max_dimensions": max(n_dims),
    }


def item_stability(
    boot_result: "BootEGAResult",
    var_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute item stability from bootstrap results.

    Parameters
    ----------
    boot_result : BootEGAResult
        Bootstrap EGA results
    var_names : list, optional
        Variable names

    Returns
    -------
    pd.DataFrame
        Item stability metrics
    """
    if len(boot_result.boot_results) == 0:
        return pd.DataFrame()

    n_vars = len(boot_result.boot_results[0].wc)
    n_boots = len(boot_result.boot_results)

    if var_names is None:
        var_names = [f"V{i+1}" for i in range(n_vars)]

    stability_data = []

    for i, name in enumerate(var_names):
        memberships = []
        for result in boot_result.boot_results:
            if not np.isnan(result.wc[i]):
                memberships.append(int(result.wc[i]))

        if len(memberships) == 0:
            stability_data.append({
                "item": name,
                "empirical_dimension": np.nan,
                "replicate_proportion": 0.0,
                "n_assignments": 0,
            })
        else:
            counts = Counter(memberships)
            modal_dim, modal_count = counts.most_common(1)[0]

            stability_data.append({
                "item": name,
                "empirical_dimension": modal_dim,
                "replicate_proportion": modal_count / len(memberships),
                "n_assignments": len(memberships),
            })

    df = pd.DataFrame(stability_data)

    return df


def replication_matrix(
    boot_result: "BootEGAResult",
) -> np.ndarray:
    """
    Compute co-assignment replication matrix.

    Measures how often pairs of items are assigned to the same dimension.

    Parameters
    ----------
    boot_result : BootEGAResult
        Bootstrap EGA results

    Returns
    -------
    np.ndarray
        Replication matrix (proportion of co-assignments)
    """
    if len(boot_result.boot_results) == 0:
        return np.array([])

    n_vars = len(boot_result.boot_results[0].wc)
    n_boots = len(boot_result.boot_results)

    replication = np.zeros((n_vars, n_vars))
    valid_counts = np.zeros((n_vars, n_vars))

    for result in boot_result.boot_results:
        wc = result.wc
        for i in range(n_vars):
            for j in range(i, n_vars):
                if not np.isnan(wc[i]) and not np.isnan(wc[j]):
                    valid_counts[i, j] += 1
                    valid_counts[j, i] += 1
                    if wc[i] == wc[j]:
                        replication[i, j] += 1
                        replication[j, i] += 1

    valid_counts[valid_counts == 0] = 1
    replication = replication / valid_counts

    return replication


def structure_replication(
    boot_result: "BootEGAResult",
    reference_structure: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute structure replication across bootstrap samples.

    Parameters
    ----------
    boot_result : BootEGAResult
        Bootstrap EGA results
    reference_structure : np.ndarray, optional
        Reference structure to compare against (defaults to typical structure)

    Returns
    -------
    dict
        Structure replication metrics
    """
    if reference_structure is None:
        reference_structure = boot_result.ega.wc

    n_boots = len(boot_result.boot_results)
    exact_matches = 0
    adjusted_rand = []

    for result in boot_result.boot_results:
        boot_wc = result.wc

        valid_mask = ~(np.isnan(reference_structure) | np.isnan(boot_wc))

        if np.all(reference_structure[valid_mask] == boot_wc[valid_mask]):
            exact_matches += 1

        ari = _adjusted_rand_index(
            reference_structure[valid_mask],
            boot_wc[valid_mask]
        )
        adjusted_rand.append(ari)

    return {
        "exact_replication_rate": exact_matches / n_boots,
        "mean_adjusted_rand": np.mean(adjusted_rand),
        "std_adjusted_rand": np.std(adjusted_rand),
        "median_adjusted_rand": np.median(adjusted_rand),
    }


def _adjusted_rand_index(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Compute Adjusted Rand Index between two clusterings."""
    from scipy.special import comb

    labels1 = labels1.astype(int)
    labels2 = labels2.astype(int)

    n = len(labels1)

    contingency = np.zeros((labels1.max() + 1, labels2.max() + 1))
    for i in range(n):
        contingency[labels1[i], labels2[i]] += 1

    sum_comb_c = sum(comb(contingency.sum(axis=1), 2))
    sum_comb_k = sum(comb(contingency.sum(axis=0), 2))
    sum_comb = sum(comb(contingency.flatten(), 2))

    expected = (sum_comb_c * sum_comb_k) / comb(n, 2)
    max_index = (sum_comb_c + sum_comb_k) / 2
    denominator = max_index - expected

    if denominator == 0:
        return 1.0 if sum_comb == expected else 0.0

    return (sum_comb - expected) / denominator
