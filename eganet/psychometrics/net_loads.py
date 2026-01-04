"""
Network loadings computation.

Computes network loadings as an analogue to factor loadings,
representing the relationship between items and dimensions.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from eganet.utils.helpers import EGAResult


def net_loads(
    ega_result: Union["EGAResult", Dict[str, Any]],
    rotation: str = "none",
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Compute network loadings.

    Network loadings represent the standardized sum of edges within
    each community, analogous to factor loadings.

    Parameters
    ----------
    ega_result : EGAResult or dict
        EGA result containing network and community memberships
    rotation : str
        Rotation method (currently only "none" supported)
    standardize : bool
        Whether to standardize loadings

    Returns
    -------
    pd.DataFrame
        Network loadings matrix (items x dimensions)

    Notes
    -----
    Network loadings are computed as the sum of absolute network
    connections to items within each dimension, normalized by
    the node's total strength.
    """
    if hasattr(ega_result, "network"):
        network = ega_result.network
        wc = ega_result.wc
        var_names = list(ega_result.dim_variables["items"])
    else:
        network = ega_result["network"]
        wc = ega_result["wc"]
        var_names = ega_result.get("var_names", [f"V{i+1}" for i in range(len(wc))])

    n_vars = network.shape[0]
    abs_network = np.abs(network)

    valid_wc = wc[~np.isnan(wc)]
    communities = np.unique(valid_wc)
    n_communities = len(communities)

    community_map = {comm: i for i, comm in enumerate(communities)}

    loadings = np.zeros((n_vars, n_communities))

    for i in range(n_vars):
        for comm in communities:
            mask = wc == comm
            comm_idx = community_map[comm]

            sum_edges = np.sum(abs_network[i, mask])
            loadings[i, comm_idx] = sum_edges

    if standardize:
        node_strength = np.sum(abs_network, axis=1)
        node_strength[node_strength == 0] = 1

        for i in range(n_vars):
            loadings[i, :] = loadings[i, :] / node_strength[i]

        for j in range(n_communities):
            col_std = np.std(loadings[:, j])
            if col_std > 0:
                loadings[:, j] = loadings[:, j] / col_std

    loadings_df = pd.DataFrame(
        loadings,
        index=var_names,
        columns=[f"Dim_{int(c)}" for c in communities]
    )

    return loadings_df


def split_loadings(loadings: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split loadings into positive and negative components.

    Parameters
    ----------
    loadings : pd.DataFrame
        Network loadings matrix

    Returns
    -------
    dict
        Dictionary with 'positive' and 'negative' loading matrices
    """
    positive = loadings.copy()
    positive[positive < 0] = 0

    negative = loadings.copy()
    negative[negative > 0] = 0
    negative = -negative

    return {
        "positive": positive,
        "negative": negative,
    }


def dominant_loading(loadings: pd.DataFrame) -> pd.Series:
    """
    Get dominant dimension for each item.

    Parameters
    ----------
    loadings : pd.DataFrame
        Network loadings matrix

    Returns
    -------
    pd.Series
        Dominant dimension for each item
    """
    abs_loadings = loadings.abs()
    dominant = abs_loadings.idxmax(axis=1)
    return dominant


def cross_loadings(
    loadings: pd.DataFrame,
    threshold: float = 0.3
) -> pd.DataFrame:
    """
    Identify cross-loading items.

    Parameters
    ----------
    loadings : pd.DataFrame
        Network loadings matrix
    threshold : float
        Threshold for significant loading

    Returns
    -------
    pd.DataFrame
        Items with cross-loadings above threshold
    """
    abs_loadings = loadings.abs()

    cross_load_mask = abs_loadings >= threshold
    n_high_loadings = cross_load_mask.sum(axis=1)

    cross_loading_items = loadings.loc[n_high_loadings > 1]

    return cross_loading_items


def loading_similarity(
    loadings1: pd.DataFrame,
    loadings2: pd.DataFrame,
    method: str = "congruence"
) -> np.ndarray:
    """
    Compute similarity between two loading matrices.

    Parameters
    ----------
    loadings1 : pd.DataFrame
        First loading matrix
    loadings2 : pd.DataFrame
        Second loading matrix
    method : str
        Similarity method ("congruence", "correlation")

    Returns
    -------
    np.ndarray
        Similarity matrix (dimensions x dimensions)
    """
    L1 = loadings1.values
    L2 = loadings2.values

    n_dim1 = L1.shape[1]
    n_dim2 = L2.shape[1]

    similarity = np.zeros((n_dim1, n_dim2))

    for i in range(n_dim1):
        for j in range(n_dim2):
            if method == "congruence":
                numerator = np.sum(L1[:, i] * L2[:, j])
                denom = np.sqrt(np.sum(L1[:, i]**2) * np.sum(L2[:, j]**2))
                similarity[i, j] = numerator / denom if denom > 0 else 0
            else:
                similarity[i, j] = np.corrcoef(L1[:, i], L2[:, j])[0, 1]

    return similarity
