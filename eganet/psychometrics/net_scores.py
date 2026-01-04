"""
Network scores computation.

Computes dimension scores (analogous to factor scores) from EGA results.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from eganet.utils.helpers import EGAResult

from eganet.psychometrics.net_loads import net_loads


def net_scores(
    data: Union[np.ndarray, pd.DataFrame],
    ega_result: Union["EGAResult", Dict[str, Any]],
    method: str = "network",
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Compute network dimension scores.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Original data matrix
    ega_result : EGAResult or dict
        EGA result with network and memberships
    method : str
        Scoring method ("network", "mean", "sum")
    standardize : bool
        Whether to standardize scores

    Returns
    -------
    pd.DataFrame
        Dimension scores (samples x dimensions)
    """
    if isinstance(data, pd.DataFrame):
        data_arr = data.values.astype(float)
    else:
        data_arr = data.astype(float)

    if hasattr(ega_result, "wc"):
        wc = ega_result.wc
    else:
        wc = ega_result["wc"]

    valid_wc = wc[~np.isnan(wc)]
    communities = np.unique(valid_wc)
    n_communities = len(communities)
    n_samples = data_arr.shape[0]

    scores = np.zeros((n_samples, n_communities))

    if method == "mean":
        for i, comm in enumerate(communities):
            mask = wc == comm
            if np.sum(mask) > 0:
                scores[:, i] = np.nanmean(data_arr[:, mask], axis=1)

    elif method == "sum":
        for i, comm in enumerate(communities):
            mask = wc == comm
            if np.sum(mask) > 0:
                scores[:, i] = np.nansum(data_arr[:, mask], axis=1)

    else:
        loadings = net_loads(ega_result, standardize=False)
        loadings_arr = loadings.values

        for i in range(n_samples):
            row = data_arr[i]
            valid_mask = ~np.isnan(row)

            if np.any(valid_mask):
                row_valid = row[valid_mask]
                loadings_valid = loadings_arr[valid_mask]

                for j in range(n_communities):
                    load_col = loadings_valid[:, j]
                    if np.sum(np.abs(load_col)) > 0:
                        scores[i, j] = np.sum(row_valid * load_col) / np.sum(np.abs(load_col))

    if standardize:
        for j in range(n_communities):
            col = scores[:, j]
            col_mean = np.nanmean(col)
            col_std = np.nanstd(col)
            if col_std > 0:
                scores[:, j] = (col - col_mean) / col_std

    scores_df = pd.DataFrame(
        scores,
        columns=[f"Dim_{int(c)}" for c in communities]
    )

    return scores_df


def score_reliability(
    data: Union[np.ndarray, pd.DataFrame],
    ega_result: Union["EGAResult", Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute reliability (omega) for each dimension.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Original data
    ega_result : EGAResult or dict
        EGA result

    Returns
    -------
    dict
        Reliability estimate for each dimension
    """
    if isinstance(data, pd.DataFrame):
        data_arr = data.values.astype(float)
    else:
        data_arr = data.astype(float)

    if hasattr(ega_result, "wc"):
        wc = ega_result.wc
    else:
        wc = ega_result["wc"]

    valid_wc = wc[~np.isnan(wc)]
    communities = np.unique(valid_wc)

    reliabilities = {}

    for comm in communities:
        mask = wc == comm
        n_items = np.sum(mask)

        if n_items < 2:
            reliabilities[f"Dim_{int(comm)}"] = np.nan
            continue

        subset = data_arr[:, mask]

        valid_rows = ~np.any(np.isnan(subset), axis=1)
        subset_valid = subset[valid_rows]

        if len(subset_valid) < 2:
            reliabilities[f"Dim_{int(comm)}"] = np.nan
            continue

        item_vars = np.var(subset_valid, axis=0, ddof=1)
        total_var = np.var(np.sum(subset_valid, axis=1), ddof=1)

        if total_var > 0:
            alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_vars) / total_var)
        else:
            alpha = np.nan

        reliabilities[f"Dim_{int(comm)}"] = alpha

    return reliabilities
