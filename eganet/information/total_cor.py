"""
Total correlation computation.
"""

from __future__ import annotations
from typing import Union, Optional
import numpy as np


def total_cor(
    data: np.ndarray,
    method: str = "vn"
) -> float:
    """
    Compute total correlation.

    Parameters
    ----------
    data : np.ndarray
        Data matrix
    method : str
        Method ("vn" for Von Neumann, "shannon" for Shannon entropy)

    Returns
    -------
    float
        Total correlation value
    """
    if method == "vn":
        from eganet.correlation.auto import auto_correlate
        correlation, _ = auto_correlate(data)
        return total_cor_matrix(correlation)
    else:
        from eganet.information.entropy import total_correlation_entropy
        return total_correlation_entropy(data)


def total_cor_matrix(correlation: np.ndarray) -> float:
    """
    Compute total correlation from correlation matrix.

    Parameters
    ----------
    correlation : np.ndarray
        Correlation matrix

    Returns
    -------
    float
        Total correlation
    """
    from eganet.information.entropy import vn_entropy

    n = correlation.shape[0]

    total_vn = vn_entropy(correlation, normalized=False)

    independent_vn = n * vn_entropy(np.array([[1.0]]), normalized=False)

    return independent_vn - total_vn


def partial_total_cor(
    data: np.ndarray,
    structure: np.ndarray
) -> dict:
    """
    Compute partial total correlation for each community.

    Parameters
    ----------
    data : np.ndarray
        Data matrix
    structure : np.ndarray
        Community memberships

    Returns
    -------
    dict
        Total correlation per community
    """
    from eganet.correlation.auto import auto_correlate

    correlation, _ = auto_correlate(data)

    valid_structure = structure[~np.isnan(structure)]
    communities = np.unique(valid_structure)

    results = {}

    for comm in communities:
        mask = structure == comm
        n_items = np.sum(mask)

        if n_items > 1:
            submatrix = correlation[np.ix_(mask, mask)]
            tc = total_cor_matrix(submatrix)
            results[f"community_{int(comm)}"] = tc
        else:
            results[f"community_{int(comm)}"] = 0.0

    total = total_cor_matrix(correlation)
    within_total = sum(results.values())
    between = total - within_total

    results["within_total"] = within_total
    results["between"] = between
    results["total"] = total

    return results


def dual_total_cor(
    data: np.ndarray,
    structure: np.ndarray
) -> dict:
    """
    Compute dual total correlation decomposition.

    Parameters
    ----------
    data : np.ndarray
        Data matrix
    structure : np.ndarray
        Community memberships

    Returns
    -------
    dict
        Dual total correlation decomposition
    """
    from eganet.correlation.auto import auto_correlate
    from eganet.information.entropy import vn_entropy

    correlation, _ = auto_correlate(data)
    n = correlation.shape[0]

    total_vn = vn_entropy(correlation, normalized=False)

    valid_structure = structure[~np.isnan(structure)]
    communities = np.unique(valid_structure)
    n_communities = len(communities)

    community_entropies = []
    for comm in communities:
        mask = structure == comm
        if np.sum(mask) > 1:
            submatrix = correlation[np.ix_(mask, mask)]
            comm_vn = vn_entropy(submatrix, normalized=False)
            community_entropies.append(comm_vn)

    sum_community_vn = sum(community_entropies)

    binding_info = sum_community_vn - total_vn

    higher_corr = np.zeros((n_communities, n_communities))
    for i, comm_i in enumerate(communities):
        for j, comm_j in enumerate(communities):
            if i == j:
                higher_corr[i, j] = 1.0
            else:
                mask_i = structure == comm_i
                mask_j = structure == comm_j
                submatrix = correlation[np.ix_(mask_i, mask_j)]
                higher_corr[i, j] = np.mean(np.abs(submatrix))

    higher_corr = (higher_corr + higher_corr.T) / 2

    residual_tc = total_cor_matrix(higher_corr)

    return {
        "total_tc": n * np.log(n) - total_vn if n > 0 else 0,
        "binding_info": binding_info,
        "residual_tc": residual_tc,
        "sum_within_vn": sum_community_vn,
        "total_vn": total_vn,
    }
