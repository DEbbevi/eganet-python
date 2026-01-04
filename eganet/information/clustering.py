"""
Information-based clustering.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def info_cluster(
    data: np.ndarray,
    method: str = "jsd",
    linkage_method: str = "ward",
    n_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Information-based hierarchical clustering.

    Parameters
    ----------
    data : np.ndarray
        Data matrix
    method : str
        Distance method ("jsd", "mi")
    linkage_method : str
        Hierarchical clustering linkage method
    n_clusters : int, optional
        Number of clusters (auto-detected if None)

    Returns
    -------
    dict
        Clustering results
    """
    n_vars = data.shape[1]

    distance_matrix = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if method == "jsd":
                from eganet.information.jsd import jsd
                d = jsd(data[:, i], data[:, j])
            else:
                from eganet.information.entropy import mutual_information
                mi = mutual_information(data[:, i], data[:, j])
                d = 1 / (1 + mi)

            distance_matrix[i, j] = d
            distance_matrix[j, i] = d

    condensed = squareform(distance_matrix)

    Z = linkage(condensed, method=linkage_method)

    if n_clusters is None:
        n_clusters = _optimal_clusters(Z, n_vars)

    labels = fcluster(Z, n_clusters, criterion='maxclust')

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "linkage": Z,
        "distance_matrix": distance_matrix,
    }


def _optimal_clusters(Z: np.ndarray, n: int) -> int:
    """
    Find optimal number of clusters using gap statistic.

    Parameters
    ----------
    Z : np.ndarray
        Linkage matrix
    n : int
        Number of items

    Returns
    -------
    int
        Optimal number of clusters
    """
    max_clusters = min(n // 2, 10)

    best_k = 2
    max_gap = -np.inf

    heights = Z[:, 2]

    for k in range(2, max_clusters + 1):
        idx = n - k
        if idx > 0 and idx < len(heights) - 1:
            gap = heights[idx + 1] - heights[idx]
            if gap > max_gap:
                max_gap = gap
                best_k = k

    return best_k


def mutual_info_clustering(
    data: np.ndarray,
    n_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Clustering based on mutual information.

    Parameters
    ----------
    data : np.ndarray
        Data matrix
    n_clusters : int, optional
        Number of clusters

    Returns
    -------
    dict
        Clustering results
    """
    from eganet.information.entropy import mutual_information

    n_vars = data.shape[1]

    mi_matrix = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            mi = mutual_information(data[:, i], data[:, j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    mi_max = mi_matrix.max()
    if mi_max > 0:
        distance = 1 - mi_matrix / mi_max
    else:
        distance = np.ones_like(mi_matrix)

    np.fill_diagonal(distance, 0)

    condensed = squareform(distance)
    Z = linkage(condensed, method='ward')

    if n_clusters is None:
        n_clusters = _optimal_clusters(Z, n_vars)

    labels = fcluster(Z, n_clusters, criterion='maxclust')

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "mi_matrix": mi_matrix,
        "linkage": Z,
    }
