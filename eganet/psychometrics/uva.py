"""
Unique Variable Analysis (UVA).

Identifies locally dependent (redundant) variables using
weighted topological overlap.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from eganet.utils.helpers import EGAResult


def uva(
    data: Union[np.ndarray, pd.DataFrame, "EGAResult"],
    n: Optional[int] = None,
    cut: float = 0.25,
    reduce: bool = True,
    reduce_method: str = "remove",
    auto: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Unique Variable Analysis.

    Identifies redundant variables using weighted topological overlap.

    Parameters
    ----------
    data : np.ndarray, pd.DataFrame, or EGAResult
        Input data or EGA result
    n : int, optional
        Sample size
    cut : float
        WTO threshold for redundancy
    reduce : bool
        Whether to reduce redundant pairs
    reduce_method : str
        Reduction method ("remove", "mean", "sum", "latent")
    auto : bool
        Automatically determine which item to remove
    verbose : bool
        Print information

    Returns
    -------
    dict
        UVA results including redundant pairs and reduced data
    """
    from eganet.core.ega import EGA
    from eganet.utils.helpers import EGAResult, is_correlation_matrix

    if isinstance(data, EGAResult) or (isinstance(data, dict) and "network" in data):
        if hasattr(data, "network"):
            network = data.network
            correlation = data.correlation
            wc = data.wc
        else:
            network = data["network"]
            correlation = data["correlation"]
            wc = data["wc"]
        var_names = [f"V{i+1}" for i in range(network.shape[0])]
        original_data = None
    else:
        if isinstance(data, pd.DataFrame):
            var_names = list(data.columns)
            data_arr = data.values.astype(float)
        else:
            data_arr = data.astype(float)
            var_names = [f"V{i+1}" for i in range(data_arr.shape[1])]

        if is_correlation_matrix(data_arr):
            from eganet.network.estimation import network_estimation
            network = network_estimation(data_arr, n=n or 100)
            correlation = data_arr
            original_data = None
        else:
            ega_result = EGA(data_arr, n=n, verbose=verbose)
            network = ega_result.network
            correlation = ega_result.correlation
            wc = ega_result.wc
            original_data = data_arr

    wto_matrix = weighted_topological_overlap(network)

    redundant_pairs = []
    n_vars = wto_matrix.shape[0]

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if wto_matrix[i, j] >= cut:
                redundant_pairs.append({
                    "var1": var_names[i],
                    "var2": var_names[j],
                    "wto": wto_matrix[i, j],
                })

    if verbose:
        print(f"Found {len(redundant_pairs)} redundant pairs with WTO >= {cut}")

    items_to_remove = []

    if reduce and len(redundant_pairs) > 0:
        for pair in redundant_pairs:
            var1_idx = var_names.index(pair["var1"])
            var2_idx = var_names.index(pair["var2"])

            if auto:
                strength1 = np.sum(np.abs(network[var1_idx, :]))
                strength2 = np.sum(np.abs(network[var2_idx, :]))

                if strength1 < strength2:
                    items_to_remove.append(pair["var1"])
                else:
                    items_to_remove.append(pair["var2"])

        items_to_remove = list(set(items_to_remove))

        if verbose:
            print(f"Items suggested for removal: {items_to_remove}")

    reduced_data = None
    if reduce and original_data is not None and len(items_to_remove) > 0:
        keep_indices = [i for i, name in enumerate(var_names) if name not in items_to_remove]

        if reduce_method == "remove":
            reduced_data = original_data[:, keep_indices]
        elif reduce_method in ["mean", "sum"]:
            reduced_data = original_data[:, keep_indices].copy()

    return {
        "wto_matrix": wto_matrix,
        "redundant_pairs": redundant_pairs,
        "items_to_remove": items_to_remove,
        "reduced_data": reduced_data,
        "cut": cut,
        "n_redundant": len(redundant_pairs),
    }


def weighted_topological_overlap(network: np.ndarray) -> np.ndarray:
    """
    Compute Weighted Topological Overlap (WTO).

    Parameters
    ----------
    network : np.ndarray
        Network adjacency matrix

    Returns
    -------
    np.ndarray
        WTO matrix
    """
    abs_network = np.abs(network)
    np.fill_diagonal(abs_network, 0)

    n = abs_network.shape[0]
    wto = np.zeros((n, n))

    strength = np.sum(abs_network, axis=1)

    for i in range(n):
        for j in range(i + 1, n):
            shared = np.sum(np.minimum(abs_network[i, :], abs_network[j, :]))
            shared += abs_network[i, j]

            min_strength = min(strength[i], strength[j])

            if min_strength > 0:
                wto[i, j] = shared / (min_strength + 1 - abs_network[i, j])
            else:
                wto[i, j] = 0

            wto[j, i] = wto[i, j]

    return wto


def redundancy_chain(
    wto_matrix: np.ndarray,
    var_names: List[str],
    cut: float = 0.25
) -> List[List[str]]:
    """
    Find chains of redundant variables.

    Parameters
    ----------
    wto_matrix : np.ndarray
        WTO matrix
    var_names : list
        Variable names
    cut : float
        WTO threshold

    Returns
    -------
    list
        List of redundancy chains
    """
    n = len(var_names)
    visited = [False] * n
    chains = []

    def dfs(node, chain):
        visited[node] = True
        chain.append(var_names[node])

        for neighbor in range(n):
            if not visited[neighbor] and wto_matrix[node, neighbor] >= cut:
                dfs(neighbor, chain)

    for i in range(n):
        if not visited[i]:
            has_high_wto = any(
                wto_matrix[i, j] >= cut for j in range(n) if j != i
            )
            if has_high_wto:
                chain = []
                dfs(i, chain)
                if len(chain) > 1:
                    chains.append(chain)

    return chains
