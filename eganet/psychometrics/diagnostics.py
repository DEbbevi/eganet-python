"""
Item diagnostics for EGA.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from eganet.utils.helpers import EGAResult


def item_diagnostics(
    ega_result: "EGAResult",
    data: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute item-level diagnostics.

    Parameters
    ----------
    ega_result : EGAResult
        EGA result
    data : np.ndarray, optional
        Original data

    Returns
    -------
    pd.DataFrame
        Item diagnostics
    """
    network = ega_result.network
    wc = ega_result.wc
    var_names = list(ega_result.dim_variables["items"])

    n_vars = len(var_names)
    abs_network = np.abs(network)
    np.fill_diagonal(abs_network, 0)

    diagnostics = []

    for i, name in enumerate(var_names):
        strength = np.sum(abs_network[i])

        within_strength = 0
        between_strength = 0

        if not np.isnan(wc[i]):
            for j in range(n_vars):
                if i != j:
                    if wc[j] == wc[i]:
                        within_strength += abs_network[i, j]
                    else:
                        between_strength += abs_network[i, j]

        connectivity = np.sum(abs_network[i] > 0)

        if strength > 0:
            within_ratio = within_strength / strength
        else:
            within_ratio = 0

        diagnostics.append({
            "item": name,
            "dimension": wc[i] if not np.isnan(wc[i]) else np.nan,
            "strength": strength,
            "within_strength": within_strength,
            "between_strength": between_strength,
            "within_ratio": within_ratio,
            "connectivity": connectivity,
        })

    df = pd.DataFrame(diagnostics)

    return df


def flag_problematic_items(
    diagnostics: pd.DataFrame,
    strength_threshold: float = 0.1,
    within_ratio_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Flag potentially problematic items.

    Parameters
    ----------
    diagnostics : pd.DataFrame
        Item diagnostics
    strength_threshold : float
        Minimum strength threshold
    within_ratio_threshold : float
        Minimum within-community ratio

    Returns
    -------
    pd.DataFrame
        Flagged items with issues
    """
    flags = []

    for _, row in diagnostics.iterrows():
        issues = []

        if row["strength"] < strength_threshold:
            issues.append("low_strength")

        if row["within_ratio"] < within_ratio_threshold:
            issues.append("low_within_ratio")

        if row["connectivity"] < 2:
            issues.append("low_connectivity")

        if np.isnan(row["dimension"]):
            issues.append("unassigned")

        if issues:
            flags.append({
                "item": row["item"],
                "issues": ", ".join(issues),
                "strength": row["strength"],
                "within_ratio": row["within_ratio"],
            })

    return pd.DataFrame(flags)


def cross_loading_diagnostics(
    ega_result: "EGAResult",
    threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Identify cross-loading items.

    Parameters
    ----------
    ega_result : EGAResult
        EGA result
    threshold : float
        Cross-loading threshold

    Returns
    -------
    pd.DataFrame
        Cross-loading diagnostics
    """
    from eganet.psychometrics.net_loads import net_loads

    loadings = net_loads(ega_result)

    cross_items = []

    for item in loadings.index:
        row = loadings.loc[item]
        high_loadings = row[np.abs(row) >= threshold]

        if len(high_loadings) > 1:
            sorted_loadings = np.abs(row).sort_values(ascending=False)
            primary = sorted_loadings.index[0]
            secondary = sorted_loadings.index[1]

            cross_items.append({
                "item": item,
                "primary_dimension": primary,
                "primary_loading": loadings.loc[item, primary],
                "secondary_dimension": secondary,
                "secondary_loading": loadings.loc[item, secondary],
                "loading_ratio": np.abs(loadings.loc[item, secondary] / loadings.loc[item, primary])
            })

    return pd.DataFrame(cross_items)
