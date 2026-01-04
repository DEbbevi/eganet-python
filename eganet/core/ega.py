"""
Exploratory Graph Analysis (EGA).

Main function for estimating the number of dimensions in psychological data
using network psychometrics and community detection.
"""

from __future__ import annotations
from typing import Union, Optional, Literal, Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from eganet.utils.helpers import (
    EGAResult,
    ensure_dimension_names,
    get_variable_names,
    is_symmetric,
    is_correlation_matrix,
    unique_length,
)
from eganet.correlation.auto import obtain_sample_correlations
from eganet.network.estimation import network_estimation
from eganet.network.community import (
    community_detection,
    community_unidimensional,
    community_consensus,
)


NetworkModel = Literal["glasso", "tmfg"]
CommunityAlgorithm = Literal["walktrap", "louvain", "leiden"]
UniMethod = Literal["louvain", "le", "expand"]
CorrelationMethod = Literal["auto", "pearson", "spearman", "polychoric", "cosine"]
NAMethod = Literal["pairwise", "listwise"]


def EGA(
    data: Union[np.ndarray, pd.DataFrame],
    n: Optional[int] = None,
    corr: CorrelationMethod = "auto",
    na_data: NAMethod = "pairwise",
    model: NetworkModel = "glasso",
    algorithm: CommunityAlgorithm = "walktrap",
    uni_method: UniMethod = "louvain",
    plot_ega: bool = False,
    verbose: bool = False,
    **kwargs
) -> EGAResult:
    """
    Exploratory Graph Analysis.

    Estimates the number of dimensions in data using network estimation
    and community detection.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data matrix or correlation matrix
    n : int, optional
        Sample size (required if correlation matrix provided)
    corr : str
        Correlation method ("auto", "pearson", "spearman", "polychoric", "cosine")
    na_data : str
        Missing data handling ("pairwise", "listwise")
    model : str
        Network estimation method ("glasso", "tmfg")
    algorithm : str
        Community detection algorithm ("walktrap", "louvain", "leiden")
    uni_method : str
        Unidimensionality method ("louvain", "le", "expand")
    plot_ega : bool
        Whether to generate plot (not implemented)
    verbose : bool
        Print progress

    Returns
    -------
    EGAResult
        EGA results containing network, community memberships, etc.

    Examples
    --------
    >>> import numpy as np
    >>> from eganet import EGA
    >>> data = np.random.randn(100, 10)
    >>> result = EGA(data)
    >>> print(result.n_dim)
    """
    if isinstance(data, pd.DataFrame):
        var_names = list(data.columns)
        data_arr = data.values.astype(float)
    else:
        data_arr = data.astype(float)
        var_names = [f"V{i+1}" for i in range(data_arr.shape[1])]

    corr_output = obtain_sample_correlations(
        data_arr, n, corr, na_data, verbose=verbose, **kwargs
    )
    correlation_matrix = corr_output["correlation_matrix"]
    sample_size = corr_output["n"]
    corr_method = corr_output.get("method", corr)

    network = network_estimation(
        correlation_matrix,
        n=sample_size,
        corr="pearson",
        model=model,
        verbose=verbose,
        **kwargs
    )

    multidim_membership = community_detection(
        network,
        algorithm=algorithm,
        **kwargs
    )

    uni_membership = community_unidimensional(
        correlation_matrix,
        network,
        uni_method=uni_method,
        **kwargs
    )

    n_uni_communities = unique_length(uni_membership)

    if n_uni_communities == 1:
        wc = np.ones(len(var_names))
        unidimensional = True
    else:
        wc = multidim_membership.copy()
        unidimensional = False

    n_dim = unique_length(wc)

    dim_variables = pd.DataFrame({
        "items": var_names,
        "dimension": wc
    })
    dim_variables = dim_variables.sort_values("dimension").reset_index(drop=True)

    methods = {
        "model": model,
        "corr": corr_method,
        "na.data": na_data,
        "algorithm": algorithm,
        "uni.method": uni_method,
        "unidimensional": unidimensional,
    }

    result = EGAResult(
        network=network,
        wc=wc,
        n_dim=n_dim,
        correlation=correlation_matrix,
        n=sample_size,
        dim_variables=dim_variables,
        methods=methods,
    )

    from eganet.information.tefi import tefi
    try:
        result.tefi = tefi(result)["vn_entropy_fit"]
    except Exception:
        result.tefi = None

    return result


def ega_estimate(
    data: Union[np.ndarray, pd.DataFrame],
    n: Optional[int] = None,
    corr: CorrelationMethod = "auto",
    na_data: NAMethod = "pairwise",
    model: NetworkModel = "glasso",
    algorithm: CommunityAlgorithm = "walktrap",
    verbose: bool = False,
    **kwargs
) -> EGAResult:
    """
    Estimate EGA without unidimensionality check.

    This is a faster version of EGA that skips the unidimensionality
    check. Useful for bootstrap iterations.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data matrix or correlation matrix
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
    verbose : bool
        Print progress

    Returns
    -------
    EGAResult
        EGA results
    """
    if isinstance(data, pd.DataFrame):
        var_names = list(data.columns)
        data_arr = data.values.astype(float)
    else:
        data_arr = data.astype(float)
        var_names = [f"V{i+1}" for i in range(data_arr.shape[1])]

    corr_output = obtain_sample_correlations(
        data_arr, n, corr, na_data, verbose=verbose, **kwargs
    )
    correlation_matrix = corr_output["correlation_matrix"]
    sample_size = corr_output["n"]
    corr_method = corr_output.get("method", corr)

    network = network_estimation(
        correlation_matrix,
        n=sample_size,
        corr="pearson",
        model=model,
        verbose=verbose,
        **kwargs
    )

    wc = community_detection(
        network,
        algorithm=algorithm,
        **kwargs
    )

    n_dim = unique_length(wc)

    dim_variables = pd.DataFrame({
        "items": var_names,
        "dimension": wc
    })
    dim_variables = dim_variables.sort_values("dimension").reset_index(drop=True)

    methods = {
        "model": model,
        "corr": corr_method,
        "na.data": na_data,
        "algorithm": algorithm,
    }

    return EGAResult(
        network=network,
        wc=wc,
        n_dim=n_dim,
        correlation=correlation_matrix,
        n=sample_size,
        dim_variables=dim_variables,
        methods=methods,
    )


def ega_fit(
    data: Union[np.ndarray, pd.DataFrame],
    n: Optional[int] = None,
    corr: CorrelationMethod = "auto",
    na_data: NAMethod = "pairwise",
    model: NetworkModel = "glasso",
    verbose: bool = False,
    **kwargs
) -> EGAResult:
    """
    Optimize EGA using TEFI.

    Tries multiple community detection algorithms and selects
    the one with the best (lowest) TEFI.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data
    n : int, optional
        Sample size
    corr : str
        Correlation method
    na_data : str
        Missing data handling
    model : str
        Network estimation method
    verbose : bool
        Print progress

    Returns
    -------
    EGAResult
        Best EGA result by TEFI
    """
    algorithms = ["walktrap", "louvain", "leiden", "fast_greedy", "leading_eigen"]

    best_result = None
    best_tefi = np.inf

    for algo in algorithms:
        try:
            result = EGA(
                data, n, corr, na_data, model, algo,
                verbose=verbose, **kwargs
            )

            if result.tefi is not None and result.tefi < best_tefi:
                best_tefi = result.tefi
                best_result = result

        except Exception:
            continue

    if best_result is None:
        best_result = EGA(data, n, corr, na_data, model, "walktrap", verbose=verbose, **kwargs)

    return best_result
