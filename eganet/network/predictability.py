"""
Network predictability analysis.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd


def network_predictability(
    network: np.ndarray,
    original_data: np.ndarray,
    newdata: Optional[np.ndarray] = None,
    ordinal_categories: int = 7,
) -> Dict[str, Any]:
    """
    Predict new data based on network.

    Parameters
    ----------
    network : np.ndarray
        Partial correlation network
    original_data : np.ndarray
        Original data used to estimate network
    newdata : np.ndarray, optional
        New data to predict (uses original_data if None)
    ordinal_categories : int
        Max categories before variable is considered continuous

    Returns
    -------
    dict
        predictions : np.ndarray
            Predicted values
        betas : np.ndarray
            Beta coefficients from network
        results : pd.DataFrame
            Performance metrics (R2, MAE) per variable
    """
    if newdata is None:
        newdata = original_data.copy()

    original_data = np.asarray(original_data)
    newdata = np.asarray(newdata)
    network = np.asarray(network)

    n_vars = newdata.shape[1]
    n_samples = newdata.shape[0]

    categories = _data_categories(np.vstack([original_data, newdata]))

    flags = {
        "dichotomous": categories == 2,
        "polytomous": (categories > 2) & (categories <= ordinal_categories),
    }
    flags["categorical"] = flags["dichotomous"] | flags["polytomous"]
    flags["continuous"] = ~flags["categorical"]

    from eganet.correlation.auto import auto_correlate
    correlation, _ = auto_correlate(original_data)
    try:
        inverse_variances = np.diag(np.linalg.inv(correlation))
    except Exception:
        inverse_variances = np.ones(n_vars)

    inv_K = np.outer(inverse_variances, inverse_variances)
    inv_K = inv_K / np.maximum(inverse_variances[:, None], 1e-10)
    betas = network * ((inv_K + inv_K.T) / 2)

    original_means = np.nanmean(original_data, axis=0)
    original_sds = np.nanstd(original_data, axis=0, ddof=1)
    original_sds = np.maximum(original_sds, 1e-10)

    newdata_scaled = (newdata - original_means) / original_sds

    newdata_scaled = np.nan_to_num(newdata_scaled, nan=0.0)

    predictions = newdata_scaled @ betas

    for i in range(n_vars):
        if flags["categorical"][i]:
            factored_data = original_data[:, i]
            thresholds = _obtain_thresholds(factored_data)
            predictions[:, i] = _cut_to_categories(predictions[:, i], thresholds)

    if np.any(flags["continuous"]):
        cont_idx = np.where(flags["continuous"])[0]
        for i in cont_idx:
            predictions[:, i] = predictions[:, i] * original_sds[i] + original_means[i]

    results = _compute_results(predictions, newdata, n_vars)

    return {
        "predictions": predictions,
        "betas": betas,
        "results": results,
        "flags": flags,
    }


def _data_categories(data: np.ndarray) -> np.ndarray:
    """Count unique values per column."""
    n_vars = data.shape[1]
    categories = np.zeros(n_vars, dtype=int)
    for i in range(n_vars):
        col = data[:, i]
        col = col[~np.isnan(col)]
        categories[i] = len(np.unique(col))
    return categories


def _obtain_thresholds(data: np.ndarray) -> np.ndarray:
    """Obtain thresholds from categorical data."""
    data = data[~np.isnan(data)]
    unique_vals = np.unique(data)
    n_cats = len(unique_vals)

    if n_cats <= 1:
        return np.array([0.0])

    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    std_val = max(std_val, 1e-10)

    scaled = (data - mean_val) / std_val

    thresholds = []
    for i in range(n_cats - 1):
        mask = data <= unique_vals[i]
        if np.any(mask):
            thresholds.append(np.max(scaled[mask]))
        else:
            thresholds.append(-np.inf)

    return np.array(thresholds)


def _cut_to_categories(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Assign continuous values to categories based on thresholds."""
    result = np.ones(len(values))
    for i, thresh in enumerate(thresholds):
        result[values > thresh] = i + 2
    return result


def _compute_results(
    predictions: np.ndarray,
    observed: np.ndarray,
    n_vars: int
) -> pd.DataFrame:
    """Compute R2 and MAE for each variable."""
    results = []
    for i in range(n_vars):
        pred = predictions[:, i]
        obs = observed[:, i]

        mask = ~np.isnan(pred) & ~np.isnan(obs)
        pred_valid = pred[mask]
        obs_valid = obs[mask]

        if len(pred_valid) > 1:
            ss_res = np.sum((obs_valid - pred_valid) ** 2)
            ss_tot = np.sum((obs_valid - np.mean(obs_valid)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r2 = max(0, min(1, r2))
            mae = np.mean(np.abs(pred_valid - obs_valid))
        else:
            r2 = np.nan
            mae = np.nan

        results.append({"R2": r2, "MAE": mae})

    return pd.DataFrame(results)


def predictability_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Summarize predictability results.

    Parameters
    ----------
    results : dict
        Output from network_predictability

    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    df = results["results"]
    summary = {
        "mean_R2": df["R2"].mean(),
        "sd_R2": df["R2"].std(),
        "min_R2": df["R2"].min(),
        "max_R2": df["R2"].max(),
        "mean_MAE": df["MAE"].mean(),
        "sd_MAE": df["MAE"].std(),
    }
    return pd.DataFrame([summary])
