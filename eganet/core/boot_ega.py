"""
Bootstrap EGA.

Provides bootstrap methods for assessing stability of EGA solutions.
"""

from __future__ import annotations
from typing import Union, Optional, Literal, Dict, Any, List
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import Counter
from joblib import Parallel, delayed

from eganet.utils.helpers import BootEGAResult, EGAResult
from eganet.core.ega import ega_estimate
from eganet.correlation.auto import obtain_sample_correlations


BootType = Literal["parametric", "resampling"]


def boot_ega(
    data: Union[np.ndarray, pd.DataFrame],
    n: Optional[int] = None,
    corr: str = "auto",
    na_data: str = "pairwise",
    model: str = "glasso",
    algorithm: str = "walktrap",
    boot_type: BootType = "resampling",
    n_boots: int = 500,
    seed: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs
) -> BootEGAResult:
    """
    Bootstrap Exploratory Graph Analysis.

    Performs bootstrap resampling to assess stability of EGA solution.

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
    boot_type : str
        Bootstrap type ("parametric", "resampling")
    n_boots : int
        Number of bootstrap iterations
    seed : int, optional
        Random seed
    n_jobs : int
        Number of parallel jobs
    verbose : bool
        Print progress

    Returns
    -------
    BootEGAResult
        Bootstrap results including stability measures
    """
    if isinstance(data, pd.DataFrame):
        data_arr = data.values.astype(float)
        var_names = list(data.columns)
    else:
        data_arr = data.astype(float)
        var_names = [f"V{i+1}" for i in range(data_arr.shape[1])]

    rng = np.random.default_rng(seed)
    n_samples, n_vars = data_arr.shape

    sample_size = n if n is not None else n_samples

    original_result = ega_estimate(
        data_arr, n, corr, na_data, model, algorithm, verbose=verbose, **kwargs
    )

    if boot_type == "parametric":
        corr_output = obtain_sample_correlations(
            data_arr, n, corr, na_data, verbose=False, **kwargs
        )
        correlation_matrix = corr_output["correlation_matrix"]

        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            from eganet.correlation.polychoric import nearest_positive_definite
            correlation_matrix = nearest_positive_definite(correlation_matrix)
            L = np.linalg.cholesky(correlation_matrix)

        def boot_iteration(i):
            iter_seed = rng.integers(0, 2**31)
            local_rng = np.random.default_rng(iter_seed)

            boot_data = local_rng.standard_normal((sample_size, n_vars)) @ L.T

            try:
                result = ega_estimate(
                    boot_data, sample_size, "pearson", na_data,
                    model, algorithm, verbose=False, **kwargs
                )
                return result
            except Exception:
                return None
    else:
        def boot_iteration(i):
            iter_seed = rng.integers(0, 2**31)
            local_rng = np.random.default_rng(iter_seed)

            indices = local_rng.choice(n_samples, size=n_samples, replace=True)
            boot_data = data_arr[indices]

            try:
                result = ega_estimate(
                    boot_data, None, corr, na_data,
                    model, algorithm, verbose=False, **kwargs
                )
                return result
            except Exception:
                return None

    if verbose:
        print(f"Running {n_boots} bootstrap iterations...")

    if n_jobs == 1:
        boot_results = []
        for i in range(n_boots):
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{n_boots}")
            result = boot_iteration(i)
            if result is not None:
                boot_results.append(result)
    else:
        boot_results = Parallel(n_jobs=n_jobs)(
            delayed(boot_iteration)(i) for i in range(n_boots)
        )
        boot_results = [r for r in boot_results if r is not None]

    if len(boot_results) == 0:
        raise ValueError("No successful bootstrap iterations")

    n_dim_counts = Counter([r.n_dim for r in boot_results])
    frequency = np.zeros(max(n_dim_counts.keys()) + 1)
    for dim, count in n_dim_counts.items():
        frequency[dim] = count / len(boot_results)

    typical_network = _compute_typical_network(boot_results)

    item_stability = _compute_item_stability(boot_results, var_names)
    dimension_stability = _compute_dimension_stability(boot_results)

    typical_ega = ega_estimate(
        typical_network, n=sample_size, corr="pearson",
        model=model, algorithm=algorithm, **kwargs
    )

    methods = {
        "boot_type": boot_type,
        "n_boots": n_boots,
        "successful_boots": len(boot_results),
        "model": model,
        "algorithm": algorithm,
    }

    return BootEGAResult(
        ega=typical_ega,
        boot_results=boot_results,
        frequency=frequency,
        typical_network=typical_network,
        item_stability=item_stability,
        dimension_stability=dimension_stability,
        methods=methods,
    )


def _compute_typical_network(boot_results: List[EGAResult]) -> np.ndarray:
    """Compute typical (median) network from bootstrap results."""
    networks = np.array([r.network for r in boot_results])
    typical = np.median(networks, axis=0)
    return typical


def _compute_item_stability(
    boot_results: List[EGAResult],
    var_names: List[str]
) -> Dict[str, float]:
    """Compute item stability across bootstrap iterations."""
    n_vars = len(var_names)
    n_boots = len(boot_results)

    stability = {}

    for i, name in enumerate(var_names):
        memberships = []
        for result in boot_results:
            if not np.isnan(result.wc[i]):
                memberships.append(result.wc[i])

        if len(memberships) == 0:
            stability[name] = 0.0
        else:
            counts = Counter(memberships)
            most_common = counts.most_common(1)[0][1]
            stability[name] = most_common / len(memberships)

    return stability


def _compute_dimension_stability(
    boot_results: List[EGAResult]
) -> Dict[int, float]:
    """Compute dimension stability across bootstrap iterations."""
    n_dim_counts = Counter([r.n_dim for r in boot_results])
    total = len(boot_results)

    stability = {}
    for dim, count in n_dim_counts.items():
        stability[dim] = count / total

    return stability


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
        Input data
    n_boots : int
        Number of bootstrap iterations
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Bootstrap ergodicity information
    """
    rng = np.random.default_rng(seed)
    n_samples = data.shape[0]

    ergo_values = []

    for i in range(n_boots):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_data = data[indices]

        try:
            result = ega_estimate(boot_data, **kwargs)
            from eganet.information.tefi import tefi
            tefi_result = tefi(result)
            ergo_values.append(tefi_result["vn_entropy_fit"])
        except Exception:
            continue

    return {
        "mean": np.mean(ergo_values),
        "std": np.std(ergo_values),
        "ci_lower": np.percentile(ergo_values, 2.5),
        "ci_upper": np.percentile(ergo_values, 97.5),
        "n_successful": len(ergo_values),
    }
