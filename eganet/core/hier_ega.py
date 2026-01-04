"""
Hierarchical EGA.

Implements hierarchical Exploratory Graph Analysis for multi-level structures.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from eganet.core.ega import EGA, ega_estimate
from eganet.information.tefi import gen_tefi


@dataclass
class HierEGAResult:
    """Result container for hierarchical EGA."""
    first_order: Any
    higher_order: Optional[Any] = None
    n_levels: int = 1
    structure: Dict[str, Any] = field(default_factory=dict)
    tefi: Optional[Dict[str, float]] = None


def hier_ega(
    data: Union[np.ndarray, pd.DataFrame],
    n: Optional[int] = None,
    corr: str = "auto",
    na_data: str = "pairwise",
    model: str = "glasso",
    algorithm: str = "walktrap",
    max_levels: int = 3,
    verbose: bool = False,
    **kwargs
) -> HierEGAResult:
    """
    Hierarchical Exploratory Graph Analysis.

    Estimates multi-level dimensionality structure.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data matrix
    n : int, optional
        Sample size
    corr : str
        Correlation method
    na_data : str
        Missing data handling
    model : str
        Network estimation method
    algorithm : str
        Community detection algorithm
    max_levels : int
        Maximum number of hierarchical levels
    verbose : bool
        Print progress

    Returns
    -------
    HierEGAResult
        Hierarchical EGA results
    """
    first_order = EGA(
        data, n, corr, na_data, model, algorithm,
        verbose=verbose, **kwargs
    )

    if first_order.n_dim <= 2:
        return HierEGAResult(
            first_order=first_order,
            higher_order=None,
            n_levels=1,
            structure={"level_1": first_order.wc},
        )

    correlation = first_order.correlation
    wc = first_order.wc

    valid_wc = wc[~np.isnan(wc)]
    communities = np.unique(valid_wc)
    n_communities = len(communities)

    higher_corr = np.zeros((n_communities, n_communities))

    for i, comm_i in enumerate(communities):
        mask_i = wc == comm_i
        for j, comm_j in enumerate(communities):
            mask_j = wc == comm_j
            if i == j:
                higher_corr[i, j] = 1.0
            else:
                submatrix = correlation[np.ix_(mask_i, mask_j)]
                higher_corr[i, j] = np.mean(np.abs(submatrix))

    higher_corr = (higher_corr + higher_corr.T) / 2
    np.fill_diagonal(higher_corr, 1.0)

    try:
        higher_order = ega_estimate(
            higher_corr, n=n or 100, corr="pearson",
            model=model, algorithm=algorithm, **kwargs
        )
    except Exception:
        higher_order = None

    if higher_order is not None and higher_order.n_dim < n_communities:
        n_levels = 2
        higher_wc = higher_order.wc

        item_to_higher = np.zeros(len(wc))
        for i, item_wc in enumerate(wc):
            if not np.isnan(item_wc):
                comm_idx = np.where(communities == item_wc)[0][0]
                item_to_higher[i] = higher_wc[comm_idx]
            else:
                item_to_higher[i] = np.nan
    else:
        n_levels = 1
        higher_order = None
        item_to_higher = None

    structure = {
        "level_1": first_order.wc,
    }
    if item_to_higher is not None:
        structure["level_2"] = item_to_higher

    tefi_result = None
    if higher_order is not None:
        try:
            tefi_result = gen_tefi(
                correlation,
                first_order.wc,
                item_to_higher
            )
        except Exception:
            pass

    return HierEGAResult(
        first_order=first_order,
        higher_order=higher_order,
        n_levels=n_levels,
        structure=structure,
        tefi=tefi_result,
    )


def hierarchical_loadings(
    hier_result: HierEGAResult,
) -> Dict[str, pd.DataFrame]:
    """
    Compute loadings for hierarchical structure.

    Parameters
    ----------
    hier_result : HierEGAResult
        Hierarchical EGA result

    Returns
    -------
    dict
        Loadings at each level
    """
    from eganet.psychometrics.net_loads import net_loads

    loadings = {}

    loadings["first_order"] = net_loads(hier_result.first_order)

    if hier_result.higher_order is not None:
        loadings["higher_order"] = net_loads(hier_result.higher_order)

    return loadings
