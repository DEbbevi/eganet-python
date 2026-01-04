"""
Invariance testing for EGA.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List
import numpy as np
import pandas as pd

from eganet.core.ega import EGA
from eganet.network.comparison import network_compare, structure_similarity


def invariance(
    data: Union[np.ndarray, pd.DataFrame],
    groups: np.ndarray,
    corr: str = "auto",
    model: str = "glasso",
    algorithm: str = "walktrap",
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Test configural and metric invariance across groups.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data
    groups : np.ndarray
        Group membership vector
    corr : str
        Correlation method
    model : str
        Network estimation method
    algorithm : str
        Community detection algorithm
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Invariance test results
    """
    if isinstance(data, pd.DataFrame):
        data_arr = data.values.astype(float)
    else:
        data_arr = data.astype(float)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    group_results = {}

    for group in unique_groups:
        mask = groups == group
        group_data = data_arr[mask]

        result = EGA(
            group_data, corr=corr, model=model,
            algorithm=algorithm, verbose=verbose, **kwargs
        )
        group_results[group] = result

    configural = _test_configural(group_results)

    metric = _test_metric(group_results)

    return {
        "group_results": group_results,
        "configural": configural,
        "metric": metric,
        "n_groups": n_groups,
    }


def _test_configural(group_results: Dict) -> Dict[str, Any]:
    """
    Test configural invariance.

    Same number of dimensions and similar community structure.
    """
    n_dims = [r.n_dim for r in group_results.values()]

    same_n_dims = len(set(n_dims)) == 1

    groups = list(group_results.keys())
    structure_sims = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            wc1 = group_results[groups[i]].wc
            wc2 = group_results[groups[j]].wc

            sim = structure_similarity(wc1, wc2)
            structure_sims.append(sim)

    if structure_sims:
        mean_ari = np.mean([s["adjusted_rand_index"] for s in structure_sims])
        mean_nmi = np.mean([s["normalized_mutual_info"] for s in structure_sims])
    else:
        mean_ari = 1.0
        mean_nmi = 1.0

    configural_supported = same_n_dims and mean_ari > 0.8

    return {
        "same_n_dimensions": same_n_dims,
        "n_dimensions_per_group": dict(zip(group_results.keys(), n_dims)),
        "mean_adjusted_rand": mean_ari,
        "mean_normalized_mutual_info": mean_nmi,
        "configural_supported": configural_supported,
    }


def _test_metric(group_results: Dict) -> Dict[str, Any]:
    """
    Test metric invariance.

    Similar network edge weights across groups.
    """
    groups = list(group_results.keys())
    network_comps = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            net1 = group_results[groups[i]].network
            net2 = group_results[groups[j]].network

            comp = network_compare(net1, net2)
            network_comps.append(comp)

    if network_comps:
        mean_corr = np.mean([c["correlation"] for c in network_comps])
        mean_frob = np.mean([c["frobenius"] for c in network_comps])
    else:
        mean_corr = 1.0
        mean_frob = 0.0

    metric_supported = mean_corr > 0.9

    return {
        "mean_network_correlation": mean_corr,
        "mean_frobenius_distance": mean_frob,
        "metric_supported": metric_supported,
    }


def measurement_invariance_bootstrap(
    data: np.ndarray,
    groups: np.ndarray,
    n_boots: int = 100,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Bootstrap test for measurement invariance.

    Parameters
    ----------
    data : np.ndarray
        Input data
    groups : np.ndarray
        Group membership
    n_boots : int
        Number of bootstrap samples
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Bootstrap invariance results
    """
    rng = np.random.default_rng(seed)

    observed = invariance(data, groups, **kwargs)
    observed_ari = observed["configural"]["mean_adjusted_rand"]

    null_aris = []

    for _ in range(n_boots):
        perm_groups = rng.permutation(groups)
        boot_result = invariance(data, perm_groups, **kwargs)
        null_aris.append(boot_result["configural"]["mean_adjusted_rand"])

    null_aris = np.array(null_aris)
    p_value = np.mean(null_aris >= observed_ari)

    return {
        "observed_ari": observed_ari,
        "p_value": p_value,
        "null_mean": np.mean(null_aris),
        "null_std": np.std(null_aris),
        "significant": p_value < 0.05,
    }
