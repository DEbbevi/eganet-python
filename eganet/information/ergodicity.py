"""
Ergodicity information metrics.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List
import numpy as np

from eganet.information.jsd import jsd, jsd_ergodicity


def ergo_info(
    individual_results: List,
    population_result,
) -> Dict[str, Any]:
    """
    Compute ergodicity information.

    Parameters
    ----------
    individual_results : list
        List of individual EGA results
    population_result : EGAResult
        Population-level EGA result

    Returns
    -------
    dict
        Ergodicity information metrics
    """
    pop_wc = population_result.wc

    individual_wcs = []
    for result in individual_results:
        if result is not None and hasattr(result, "wc"):
            individual_wcs.append(result.wc)

    if len(individual_wcs) == 0:
        return {
            "ergodicity_index": np.nan,
            "mean_jsd": np.nan,
            "n_individuals": 0,
        }

    jsd_values = []
    for ind_wc in individual_wcs:
        min_len = min(len(pop_wc), len(ind_wc))

        pop_valid = pop_wc[:min_len][~np.isnan(pop_wc[:min_len])]
        ind_valid = ind_wc[:min_len][~np.isnan(ind_wc[:min_len])]

        if len(pop_valid) > 0 and len(ind_valid) > 0:
            jsd_val = jsd(pop_valid, ind_valid)
            jsd_values.append(jsd_val)

    if len(jsd_values) == 0:
        return {
            "ergodicity_index": np.nan,
            "mean_jsd": np.nan,
            "n_individuals": 0,
        }

    return {
        "ergodicity_index": 1 - np.mean(jsd_values),
        "mean_jsd": np.mean(jsd_values),
        "std_jsd": np.std(jsd_values),
        "median_jsd": np.median(jsd_values),
        "n_individuals": len(jsd_values),
        "jsd_values": jsd_values,
    }


def boot_ergo_info(
    data: np.ndarray,
    n_boots: int = 500,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Bootstrap ergodicity information.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    n_boots : int
        Number of bootstrap samples
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Bootstrap ergodicity information
    """
    from eganet.core.dyn_ega import dyn_ega

    rng = np.random.default_rng(seed)
    n_time = data.shape[0]

    ergo_values = []

    for _ in range(n_boots):
        indices = rng.choice(n_time, size=n_time, replace=True)
        boot_data = data[indices]

        try:
            result = dyn_ega(boot_data, **kwargs)

            if result.population is not None:
                from eganet.information.tefi import tefi
                tefi_result = tefi(result.population)
                ergo_values.append(tefi_result["vn_entropy_fit"])
        except Exception:
            continue

    if len(ergo_values) == 0:
        return {
            "mean": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    return {
        "mean": np.mean(ergo_values),
        "std": np.std(ergo_values),
        "ci_lower": np.percentile(ergo_values, 2.5),
        "ci_upper": np.percentile(ergo_values, 97.5),
        "n_successful": len(ergo_values),
    }
