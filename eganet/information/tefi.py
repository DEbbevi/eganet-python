"""
Total Entropy Fit Index (TEFI).

Implements TEFI for evaluating the fit of EGA community structures.
"""

from __future__ import annotations
from typing import Union, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

from eganet.information.entropy import vn_entropy

if TYPE_CHECKING:
    from eganet.utils.helpers import EGAResult


def tefi(
    ega_result: Union["EGAResult", Dict[str, Any]],
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute Total Entropy Fit Index.

    Parameters
    ----------
    ega_result : EGAResult or dict
        EGA result containing network, wc (memberships), and correlation
    verbose : bool
        Print information

    Returns
    -------
    dict
        Dictionary with TEFI components

    Notes
    -----
    TEFI is based on Von Neumann entropy and measures how well the
    community structure captures the correlation structure.

    Lower values indicate better fit.
    """
    if hasattr(ega_result, "network"):
        correlation = ega_result.correlation
        wc = ega_result.wc
        network = ega_result.network
    else:
        correlation = ega_result["correlation"]
        wc = ega_result["wc"]
        network = ega_result.get("network", None)

    n = correlation.shape[0]

    valid_wc = wc[~np.isnan(wc)]
    communities = np.unique(valid_wc)
    n_communities = len(communities)

    if n_communities == 0:
        return {
            "vn_entropy_fit": np.nan,
            "total_vn_entropy": np.nan,
            "average_within_entropy": np.nan,
            "n_communities": 0,
        }

    total_vn_entropy = vn_entropy(correlation, normalized=False)

    within_entropies = []

    for comm in communities:
        mask = wc == comm
        n_items = np.sum(mask)

        if n_items > 1:
            submatrix = correlation[np.ix_(mask, mask)]
            comm_entropy = vn_entropy(submatrix, normalized=False)
            within_entropies.append(comm_entropy)
        else:
            within_entropies.append(0)

    average_within_entropy = np.mean(within_entropies) if within_entropies else 0

    vn_entropy_fit = total_vn_entropy - np.sum(within_entropies)

    if n > 1:
        max_entropy = np.log(n)
        if max_entropy > 0:
            vn_entropy_fit_normalized = vn_entropy_fit / max_entropy
        else:
            vn_entropy_fit_normalized = vn_entropy_fit
    else:
        vn_entropy_fit_normalized = vn_entropy_fit

    return {
        "vn_entropy_fit": vn_entropy_fit,
        "total_vn_entropy": total_vn_entropy,
        "within_entropies": within_entropies,
        "average_within_entropy": average_within_entropy,
        "vn_entropy_fit_normalized": vn_entropy_fit_normalized,
        "n_communities": n_communities,
    }


def gen_tefi(
    correlation: np.ndarray,
    structure: np.ndarray,
    higher_order: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Generalized TEFI for hierarchical structures.

    Parameters
    ----------
    correlation : np.ndarray
        Correlation matrix
    structure : np.ndarray
        First-order community memberships
    higher_order : np.ndarray, optional
        Higher-order community memberships

    Returns
    -------
    dict
        Generalized TEFI components
    """
    first_order_tefi = tefi({
        "correlation": correlation,
        "wc": structure,
    })

    if higher_order is None:
        return first_order_tefi

    n = correlation.shape[0]
    valid_structure = structure[~np.isnan(structure)]
    communities = np.unique(valid_structure)
    n_first = len(communities)

    higher_corr = np.zeros((n_first, n_first))

    for i, comm_i in enumerate(communities):
        mask_i = structure == comm_i
        for j, comm_j in enumerate(communities):
            mask_j = structure == comm_j
            if i == j:
                higher_corr[i, j] = 1.0
            else:
                submatrix = correlation[np.ix_(mask_i, mask_j)]
                higher_corr[i, j] = np.mean(submatrix)

    higher_corr = (higher_corr + higher_corr.T) / 2

    valid_higher = higher_order[~np.isnan(higher_order)]
    higher_communities = np.unique(valid_higher)

    higher_entropy = vn_entropy(higher_corr, normalized=False)

    within_higher_entropy = 0
    for h_comm in higher_communities:
        h_mask = higher_order == h_comm
        if np.sum(h_mask) > 1:
            h_submatrix = higher_corr[np.ix_(h_mask, h_mask)]
            within_higher_entropy += vn_entropy(h_submatrix, normalized=False)

    higher_tefi_value = higher_entropy - within_higher_entropy

    return {
        "first_order_tefi": first_order_tefi["vn_entropy_fit"],
        "higher_order_tefi": higher_tefi_value,
        "total_tefi": first_order_tefi["vn_entropy_fit"] + higher_tefi_value,
        "n_first_order": n_first,
        "n_higher_order": len(higher_communities),
    }


def tefi_compare(
    results: list,
    names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Compare TEFI across multiple EGA solutions.

    Parameters
    ----------
    results : list
        List of EGA results or dicts with correlation and wc
    names : list, optional
        Names for each result

    Returns
    -------
    dict
        Comparison of TEFI values
    """
    if names is None:
        names = [f"Model_{i+1}" for i in range(len(results))]

    tefi_values = []
    n_dims = []

    for result in results:
        tefi_result = tefi(result)
        tefi_values.append(tefi_result["vn_entropy_fit"])

        if hasattr(result, "n_dim"):
            n_dims.append(result.n_dim)
        else:
            wc = result.get("wc", result.get("memberships", np.array([])))
            valid = wc[~np.isnan(wc)]
            n_dims.append(len(np.unique(valid)))

    best_idx = np.argmin(tefi_values)

    return {
        "names": names,
        "tefi_values": tefi_values,
        "n_dimensions": n_dims,
        "best_model": names[best_idx],
        "best_tefi": tefi_values[best_idx],
    }
