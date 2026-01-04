"""
Network comparison methods.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List
import numpy as np

from eganet.network.metrics import frobenius_norm


def network_compare(
    network1: np.ndarray,
    network2: np.ndarray,
    method: str = "all"
) -> Dict[str, float]:
    """
    Compare two networks.

    Parameters
    ----------
    network1 : np.ndarray
        First network
    network2 : np.ndarray
        Second network
    method : str
        Comparison method ("frobenius", "correlation", "edge", "all")

    Returns
    -------
    dict
        Comparison metrics
    """
    results = {}

    if method in ["frobenius", "all"]:
        results["frobenius"] = frobenius_norm(network1, network2)

    if method in ["correlation", "all"]:
        lower1 = network1[np.tril_indices_from(network1, k=-1)]
        lower2 = network2[np.tril_indices_from(network2, k=-1)]
        results["correlation"] = np.corrcoef(lower1, lower2)[0, 1]

    if method in ["edge", "all"]:
        edges1 = np.abs(network1) > 0
        edges2 = np.abs(network2) > 0
        np.fill_diagonal(edges1, False)
        np.fill_diagonal(edges2, False)

        intersection = np.sum(edges1 & edges2)
        union = np.sum(edges1 | edges2)

        results["jaccard"] = intersection / union if union > 0 else 0

        true_positive = np.sum(edges1 & edges2)
        false_positive = np.sum(~edges1 & edges2)
        false_negative = np.sum(edges1 & ~edges2)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        results["precision"] = precision
        results["recall"] = recall
        results["f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return results


def permutation_test(
    network1: np.ndarray,
    network2: np.ndarray,
    n_permutations: int = 1000,
    statistic: str = "frobenius",
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Permutation test for network difference.

    Parameters
    ----------
    network1 : np.ndarray
        First network
    network2 : np.ndarray
        Second network
    n_permutations : int
        Number of permutations
    statistic : str
        Test statistic
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Test results including p-value
    """
    rng = np.random.default_rng(seed)

    if statistic == "frobenius":
        observed = frobenius_norm(network1, network2)
    else:
        lower1 = network1[np.tril_indices_from(network1, k=-1)]
        lower2 = network2[np.tril_indices_from(network2, k=-1)]
        observed = 1 - np.corrcoef(lower1, lower2)[0, 1]

    combined = np.stack([network1, network2])
    null_distribution = []

    for _ in range(n_permutations):
        perm = rng.permutation(2)
        perm_networks = combined[perm]

        if statistic == "frobenius":
            null_stat = frobenius_norm(perm_networks[0], perm_networks[1])
        else:
            lower1 = perm_networks[0][np.tril_indices_from(network1, k=-1)]
            lower2 = perm_networks[1][np.tril_indices_from(network2, k=-1)]
            null_stat = 1 - np.corrcoef(lower1, lower2)[0, 1]

        null_distribution.append(null_stat)

    null_distribution = np.array(null_distribution)
    p_value = np.mean(null_distribution >= observed)

    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": np.mean(null_distribution),
        "null_std": np.std(null_distribution),
    }


def edge_wise_comparison(
    network1: np.ndarray,
    network2: np.ndarray,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Compare networks edge by edge.

    Parameters
    ----------
    network1 : np.ndarray
        First network
    network2 : np.ndarray
        Second network
    threshold : float
        Threshold for significant difference

    Returns
    -------
    dict
        Edge-wise comparison results
    """
    diff = network1 - network2

    significant_edges = []
    n = network1.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(diff[i, j]) > threshold:
                significant_edges.append({
                    "i": i,
                    "j": j,
                    "weight1": network1[i, j],
                    "weight2": network2[i, j],
                    "difference": diff[i, j],
                })

    return {
        "n_different": len(significant_edges),
        "significant_edges": significant_edges,
        "mean_abs_diff": np.mean(np.abs(diff[np.triu_indices(n, k=1)])),
        "max_diff": np.max(np.abs(diff)),
    }


def structure_similarity(
    wc1: np.ndarray,
    wc2: np.ndarray,
) -> Dict[str, float]:
    """
    Compare community structures.

    Parameters
    ----------
    wc1 : np.ndarray
        First community memberships
    wc2 : np.ndarray
        Second community memberships

    Returns
    -------
    dict
        Structure similarity metrics
    """
    valid_mask = ~(np.isnan(wc1) | np.isnan(wc2))
    wc1_valid = wc1[valid_mask].astype(int)
    wc2_valid = wc2[valid_mask].astype(int)

    exact_match = np.all(wc1_valid == wc2_valid)

    from eganet.psychometrics.stability import _adjusted_rand_index
    ari = _adjusted_rand_index(wc1_valid, wc2_valid)

    n = len(wc1_valid)
    contingency = np.zeros((wc1_valid.max() + 1, wc2_valid.max() + 1))
    for i in range(n):
        contingency[wc1_valid[i], wc2_valid[i]] += 1

    row_sum = contingency.sum(axis=1)
    col_sum = contingency.sum(axis=0)
    total = contingency.sum()

    mi = 0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            if contingency[i, j] > 0:
                mi += contingency[i, j] / total * np.log(
                    contingency[i, j] * total / (row_sum[i] * col_sum[j])
                )

    h1 = -np.sum(row_sum / total * np.log(row_sum / total + 1e-10))
    h2 = -np.sum(col_sum / total * np.log(col_sum / total + 1e-10))
    nmi = 2 * mi / (h1 + h2) if (h1 + h2) > 0 else 0

    return {
        "exact_match": exact_match,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "n_communities_1": len(np.unique(wc1_valid)),
        "n_communities_2": len(np.unique(wc2_valid)),
    }
