"""
Dynamic EGA.

Implements dynamic Exploratory Graph Analysis for time series data.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from eganet.core.ega import EGA, ega_estimate
from eganet.utils.embedding import embed


@dataclass
class DynEGAResult:
    """Result container for dynamic EGA."""
    population: Any
    individual: Optional[Dict[str, Any]] = None
    group: Optional[Dict[str, Any]] = None
    dynamics: Optional[Dict[str, Any]] = None
    methods: Dict[str, Any] = field(default_factory=dict)


def dyn_ega(
    data: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
    n_embed: int = 5,
    tau: int = 1,
    delta: int = 1,
    level: str = "population",
    model: str = "glasso",
    algorithm: str = "walktrap",
    verbose: bool = False,
    **kwargs
) -> DynEGAResult:
    """
    Dynamic Exploratory Graph Analysis.

    Estimates dynamic dimensionality from time series data.

    Parameters
    ----------
    data : np.ndarray, pd.DataFrame, or list
        Time series data (samples x variables) or list of individual time series
    n_embed : int
        Embedding dimension
    tau : int
        Time delay for embedding
    delta : int
        Derivative order (0=raw, 1=velocity, 2=acceleration)
    level : str
        Analysis level ("population", "individual", "group")
    model : str
        Network estimation method
    algorithm : str
        Community detection algorithm
    verbose : bool
        Print progress

    Returns
    -------
    DynEGAResult
        Dynamic EGA results
    """
    if isinstance(data, list):
        individual_data = data
        is_multi_individual = True
    else:
        if isinstance(data, pd.DataFrame):
            data_arr = data.values.astype(float)
        else:
            data_arr = data.astype(float)
        individual_data = [data_arr]
        is_multi_individual = False

    embedded_data = []
    for ind_data in individual_data:
        embedded = embed(ind_data, n_embed=n_embed, tau=tau)

        if delta > 0:
            embedded = _compute_derivatives(embedded, delta)

        embedded_data.append(embedded)

    if level in ["population", "group"]:
        all_embedded = np.vstack(embedded_data)

        population_result = EGA(
            all_embedded, model=model, algorithm=algorithm,
            verbose=verbose, **kwargs
        )
    else:
        population_result = None

    individual_results = None
    if level == "individual" or is_multi_individual:
        individual_results = {}
        for i, emb_data in enumerate(embedded_data):
            if emb_data.shape[0] > emb_data.shape[1]:
                try:
                    ind_result = ega_estimate(
                        emb_data, model=model, algorithm=algorithm, **kwargs
                    )
                    individual_results[f"individual_{i+1}"] = ind_result
                except Exception:
                    individual_results[f"individual_{i+1}"] = None

    group_result = None
    if level == "group" and is_multi_individual:
        pass

    methods = {
        "n_embed": n_embed,
        "tau": tau,
        "delta": delta,
        "level": level,
        "model": model,
        "algorithm": algorithm,
    }

    return DynEGAResult(
        population=population_result,
        individual=individual_results,
        group=group_result,
        methods=methods,
    )


def _compute_derivatives(data: np.ndarray, order: int = 1) -> np.ndarray:
    """Compute time derivatives of embedded data."""
    result = data.copy()

    for _ in range(order):
        result = np.diff(result, axis=0)

    return result


def dyn_ega_ind_pop(
    data: List[np.ndarray],
    n_embed: int = 5,
    tau: int = 1,
    model: str = "glasso",
    algorithm: str = "walktrap",
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Individual and population level dynamic EGA.

    Parameters
    ----------
    data : list
        List of individual time series
    n_embed : int
        Embedding dimension
    tau : int
        Time delay
    model : str
        Network estimation method
    algorithm : str
        Community detection algorithm
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Individual and population EGA results
    """
    individual_results = dyn_ega(
        data, n_embed, tau, level="individual",
        model=model, algorithm=algorithm, verbose=verbose, **kwargs
    )

    population_results = dyn_ega(
        data, n_embed, tau, level="population",
        model=model, algorithm=algorithm, verbose=verbose, **kwargs
    )

    return {
        "individual": individual_results,
        "population": population_results,
    }


def ergo_info(
    data: List[np.ndarray],
    n_embed: int = 5,
    **kwargs
) -> Dict[str, float]:
    """
    Compute ergodicity information index.

    Measures similarity between individual and population dynamics.

    Parameters
    ----------
    data : list
        List of individual time series
    n_embed : int
        Embedding dimension

    Returns
    -------
    dict
        Ergodicity information metrics
    """
    from eganet.information.jsd import jsd

    results = dyn_ega_ind_pop(data, n_embed, **kwargs)

    pop_wc = results["population"].population.wc

    individual_jsd = []

    for key, ind_result in results["individual"].individual.items():
        if ind_result is not None:
            ind_wc = ind_result.wc

            min_len = min(len(pop_wc), len(ind_wc))
            jsd_val = jsd(
                pop_wc[:min_len],
                ind_wc[:min_len]
            )
            individual_jsd.append(jsd_val)

    if len(individual_jsd) == 0:
        return {
            "mean_jsd": np.nan,
            "ergodicity_index": np.nan,
        }

    mean_jsd = np.mean(individual_jsd)

    ergodicity_index = 1 - mean_jsd

    return {
        "mean_jsd": mean_jsd,
        "std_jsd": np.std(individual_jsd),
        "ergodicity_index": ergodicity_index,
        "n_individuals": len(individual_jsd),
    }
