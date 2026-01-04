"""
Simulate Exploratory Graphical Model (EGM).

Generate data from network models with specified structure.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List
import numpy as np


def sim_egm(
    communities: Union[int, List[int]],
    variables_per_community: Union[int, List[int]] = 5,
    n: int = 500,
    p_in: float = 0.7,
    p_out: float = 0.1,
    loading_range: tuple = (0.4, 0.7),
    cross_loading_range: tuple = (0.0, 0.2),
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simulate data from an Exploratory Graphical Model.

    Parameters
    ----------
    communities : int or list
        Number of communities or list of community sizes
    variables_per_community : int or list
        Number of variables per community
    n : int
        Sample size
    p_in : float
        Within-community edge probability
    p_out : float
        Between-community edge probability
    loading_range : tuple
        Range for within-community loadings
    cross_loading_range : tuple
        Range for cross-loadings
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Simulated data, network, and true structure
    """
    rng = np.random.default_rng(seed)

    if isinstance(communities, int):
        n_communities = communities
        vars_per_comm = [variables_per_community] * n_communities
    else:
        n_communities = len(communities)
        vars_per_comm = communities

    n_vars = sum(vars_per_comm)

    true_wc = np.zeros(n_vars)
    idx = 0
    for comm, n_comm_vars in enumerate(vars_per_comm, start=1):
        true_wc[idx:idx + n_comm_vars] = comm
        idx += n_comm_vars

    network = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            same_community = true_wc[i] == true_wc[j]

            if same_community:
                if rng.random() < p_in:
                    weight = rng.uniform(*loading_range)
                    network[i, j] = weight
                    network[j, i] = weight
            else:
                if rng.random() < p_out:
                    weight = rng.uniform(*cross_loading_range)
                    network[i, j] = weight
                    network[j, i] = weight

    correlation = network.copy()
    np.fill_diagonal(correlation, 1.0)

    eigenvalues = np.linalg.eigvalsh(correlation)
    if np.min(eigenvalues) < 0:
        correlation = correlation - np.min(eigenvalues) * np.eye(n_vars) + 0.01 * np.eye(n_vars)
        diag = np.sqrt(np.diag(correlation))
        correlation = correlation / np.outer(diag, diag)

    try:
        L = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        from eganet.correlation.polychoric import nearest_positive_definite
        correlation = nearest_positive_definite(correlation)
        L = np.linalg.cholesky(correlation)

    data = rng.standard_normal((n, n_vars)) @ L.T

    return {
        "data": data,
        "network": network,
        "correlation": correlation,
        "true_wc": true_wc,
        "n_communities": n_communities,
        "n_variables": n_vars,
    }


def sim_hierarchical_egm(
    first_order_communities: int = 4,
    higher_order_communities: int = 2,
    variables_per_community: int = 5,
    n: int = 500,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simulate hierarchical EGM structure.

    Parameters
    ----------
    first_order_communities : int
        Number of first-order communities
    higher_order_communities : int
        Number of higher-order communities
    variables_per_community : int
        Variables per first-order community
    n : int
        Sample size
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Simulated hierarchical data
    """
    rng = np.random.default_rng(seed)

    first_per_higher = first_order_communities // higher_order_communities

    higher_wc = np.zeros(first_order_communities)
    for i in range(first_order_communities):
        higher_wc[i] = (i // first_per_higher) + 1

    sim_result = sim_egm(
        communities=first_order_communities,
        variables_per_community=variables_per_community,
        n=n,
        p_in=0.7,
        p_out=0.3,
        seed=seed,
    )

    item_higher_wc = np.zeros(sim_result["n_variables"])
    for i, wc in enumerate(sim_result["true_wc"]):
        first_comm_idx = int(wc) - 1
        item_higher_wc[i] = higher_wc[first_comm_idx]

    sim_result["higher_order_wc"] = item_higher_wc
    sim_result["first_order_higher_wc"] = higher_wc

    return sim_result
