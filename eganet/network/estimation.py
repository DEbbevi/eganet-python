"""
Network estimation methods.

Implements network estimation methods:
- GLASSO (Graphical LASSO) with EBIC model selection
- TMFG (Triangulated Maximally Filtered Graph)
"""

from __future__ import annotations
from typing import Union, Optional, Literal, Dict, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from eganet.correlation.auto import obtain_sample_correlations


NetworkModel = Literal["glasso", "tmfg"]


@dataclass
class NetworkEstimationResult:
    """Result from network estimation."""
    network: np.ndarray
    methods: Dict[str, Any]
    correlation: Optional[np.ndarray] = None
    n: Optional[int] = None


def glasso(
    data: Union[np.ndarray, pd.DataFrame],
    n: Optional[int] = None,
    corr: str = "auto",
    na_data: str = "pairwise",
    gamma: float = 0.5,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    model_selection: str = "ebic",
    verbose: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Estimate sparse network using Graphical LASSO with EBIC model selection.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data or correlation matrix
    n : int, optional
        Sample size (required if correlation matrix provided)
    corr : str
        Correlation method
    na_data : str
        Missing data handling
    gamma : float
        EBIC tuning parameter (0 to 0.5)
    n_lambda : int
        Number of lambda values to try
    lambda_min_ratio : float
        Minimum lambda as ratio of maximum
    model_selection : str
        Model selection criterion ("ebic" or "bic")
    verbose : bool
        Print progress

    Returns
    -------
    np.ndarray
        Estimated network (partial correlation matrix)
    """
    output = obtain_sample_correlations(
        data, n, corr, na_data, verbose=verbose, **kwargs
    )
    correlation_matrix = output["correlation_matrix"]
    sample_size = output["n"]

    n_vars = correlation_matrix.shape[0]

    from sklearn.covariance import GraphicalLassoCV, graphical_lasso

    lambda_max = np.max(np.abs(correlation_matrix - np.eye(n_vars)))
    lambda_min = lambda_max * lambda_min_ratio
    alphas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_lambda)

    best_score = -np.inf
    best_network = np.zeros((n_vars, n_vars))
    best_lambda = alphas[0]

    for alpha in alphas:
        try:
            precision, _ = graphical_lasso(
                correlation_matrix,
                alpha=alpha,
                max_iter=500,
                mode='cd'
            )

            diagonal = np.sqrt(np.diag(precision))
            diagonal[diagonal == 0] = 1
            partial_corr = -precision / np.outer(diagonal, diagonal)
            np.fill_diagonal(partial_corr, 0)

            n_edges = np.sum(np.abs(partial_corr) > 1e-10) / 2

            if model_selection == "ebic":
                log_like = _log_likelihood(correlation_matrix, precision, sample_size)
                score = _ebic(log_like, n_vars, n_edges, sample_size, gamma)
            else:
                log_like = _log_likelihood(correlation_matrix, precision, sample_size)
                score = _bic(log_like, n_vars, n_edges, sample_size)

            if score > best_score:
                best_score = score
                best_network = partial_corr
                best_lambda = alpha

        except Exception:
            continue

    methods = {
        "model": "glasso",
        "model.selection": model_selection,
        "gamma": gamma,
        "lambda": best_lambda,
        "nlambda": n_lambda,
        "lambda.min.ratio": lambda_min_ratio,
    }

    network = best_network.copy()
    network[np.abs(network) < 1e-10] = 0

    return network


def _log_likelihood(S: np.ndarray, precision: np.ndarray, n: int) -> float:
    """Compute log-likelihood for Gaussian graphical model."""
    p = S.shape[0]
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        return -np.inf
    ll = n / 2 * (logdet - np.trace(S @ precision) - p * np.log(2 * np.pi))
    return ll


def _ebic(log_like: float, p: int, k: float, n: int, gamma: float) -> float:
    """Compute Extended BIC."""
    return 2 * log_like - k * np.log(n) - 4 * gamma * k * np.log(p)


def _bic(log_like: float, p: int, k: float, n: int) -> float:
    """Compute BIC."""
    return 2 * log_like - k * np.log(n)


def tmfg(
    data: Union[np.ndarray, pd.DataFrame],
    n: Optional[int] = None,
    corr: str = "auto",
    na_data: str = "pairwise",
    partial: bool = False,
    verbose: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Triangulated Maximally Filtered Graph.

    Constructs a planar network with structural constraint of 3n-6 edges.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data or correlation matrix
    n : int, optional
        Sample size
    corr : str
        Correlation method
    na_data : str
        Missing data handling
    partial : bool
        Whether to return partial correlations (LoGo method)
    verbose : bool
        Print progress

    Returns
    -------
    np.ndarray
        TMFG network (zero-order or partial correlations)
    """
    output = obtain_sample_correlations(
        data, n, corr, na_data, verbose=verbose, **kwargs
    )
    correlation_matrix = output["correlation_matrix"]

    n_nodes = correlation_matrix.shape[0]

    if n_nodes < 9 and verbose:
        print("Warning: TMFG requires more than 9 nodes for chordal property")

    absolute_matrix = np.abs(correlation_matrix)

    inserted = np.zeros(n_nodes, dtype=int)
    separator_rows = n_nodes - 4
    triangles = np.zeros((2 * n_nodes - 4, 3), dtype=int)
    separators = np.zeros((separator_rows, 3), dtype=int)

    node_strength = np.sum(absolute_matrix, axis=0)
    mean_corr = np.mean(absolute_matrix)
    four_nodes = np.sum(
        absolute_matrix * (absolute_matrix > mean_corr),
        axis=0
    )

    top_four = np.argsort(four_nodes)[::-1][:4]
    inserted[:4] = top_four

    remaining = list(set(range(n_nodes)) - set(top_four))

    triangles[0] = [inserted[1], inserted[2], inserted[3]]
    triangles[1] = [inserted[0], inserted[2], inserted[3]]
    triangles[2] = [inserted[0], inserted[1], inserted[3]]
    triangles[3] = [inserted[0], inserted[1], inserted[2]]

    network = np.eye(n_nodes)
    for i in range(4):
        for j in range(i + 1, 4):
            network[inserted[i], inserted[j]] = correlation_matrix[inserted[i], inserted[j]]
            network[inserted[j], inserted[i]] = correlation_matrix[inserted[j], inserted[i]]

    gain_columns = 2 * (n_nodes - 2)
    gain = np.full((n_nodes, gain_columns), -np.inf)

    for t in range(4):
        for node in remaining:
            gain[node, t] = np.sum(absolute_matrix[node, triangles[t]])

    triangle_count = 4

    for i in range(4, n_nodes):
        if len(remaining) == 1:
            add_vertex = remaining[0]
            max_gain_idx = np.argmax(gain[add_vertex])
        else:
            gains_remaining = gain[remaining]
            max_idx = np.unravel_index(
                np.argmax(gains_remaining),
                gains_remaining.shape
            )
            add_vertex = remaining[max_idx[0]]
            max_gain_idx = max_idx[1]

        remaining.remove(add_vertex)
        inserted[i] = add_vertex

        triangle = triangles[max_gain_idx]
        for node in triangle:
            network[add_vertex, node] = correlation_matrix[add_vertex, node]
            network[node, add_vertex] = correlation_matrix[node, add_vertex]

        separators[i - 4] = triangle

        new_tri_1 = [triangle[0], triangle[2], add_vertex]
        new_tri_2 = [triangle[1], triangle[2], add_vertex]
        triangles[max_gain_idx] = [triangle[0], triangle[1], add_vertex]
        triangles[triangle_count] = new_tri_1
        triangles[triangle_count + 1] = new_tri_2

        gain[add_vertex, :] = 0

        for node in remaining:
            gain[node, max_gain_idx] = np.sum(absolute_matrix[node, triangles[max_gain_idx]])
            gain[node, triangle_count] = np.sum(absolute_matrix[node, triangles[triangle_count]])
            gain[node, triangle_count + 1] = np.sum(absolute_matrix[node, triangles[triangle_count + 1]])

        triangle_count += 2

    cliques = np.zeros((n_nodes - 3, 4), dtype=int)
    cliques[0] = inserted[:4]
    cliques[1:] = np.column_stack([separators, inserted[4:]])

    if partial:
        network = _logo_inversion(correlation_matrix, cliques, separators)

    np.fill_diagonal(network, 0)

    return network


def _logo_inversion(
    correlation_matrix: np.ndarray,
    cliques: np.ndarray,
    separators: np.ndarray
) -> np.ndarray:
    """
    Local-Global Inversion for partial correlations.

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Original correlation matrix
    cliques : np.ndarray
        4-cliques from TMFG
    separators : np.ndarray
        3-cliques (separators) from TMFG

    Returns
    -------
    np.ndarray
        Partial correlation network
    """
    n_nodes = correlation_matrix.shape[0]
    precision = np.zeros((n_nodes, n_nodes))

    n_separators = separators.shape[0]

    for i in range(n_separators):
        clique = cliques[i]
        clique_corr = correlation_matrix[np.ix_(clique, clique)]
        try:
            clique_inv = np.linalg.inv(clique_corr)
            for ii, ci in enumerate(clique):
                for jj, cj in enumerate(clique):
                    precision[ci, cj] += clique_inv[ii, jj]
        except np.linalg.LinAlgError:
            pass

        sep = separators[i]
        sep_corr = correlation_matrix[np.ix_(sep, sep)]
        try:
            sep_inv = np.linalg.inv(sep_corr)
            for ii, si in enumerate(sep):
                for jj, sj in enumerate(sep):
                    precision[si, sj] -= sep_inv[ii, jj]
        except np.linalg.LinAlgError:
            pass

    clique = cliques[n_separators]
    clique_corr = correlation_matrix[np.ix_(clique, clique)]
    try:
        clique_inv = np.linalg.inv(clique_corr)
        for ii, ci in enumerate(clique):
            for jj, cj in enumerate(clique):
                precision[ci, cj] += clique_inv[ii, jj]
    except np.linalg.LinAlgError:
        pass

    diagonal = np.sqrt(np.diag(precision))
    diagonal[diagonal == 0] = 1
    partial_corr = -precision / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial_corr, 0)

    return partial_corr


def network_estimation(
    data: Union[np.ndarray, pd.DataFrame],
    n: Optional[int] = None,
    corr: str = "auto",
    na_data: str = "pairwise",
    model: NetworkModel = "glasso",
    network_only: bool = True,
    verbose: bool = False,
    **kwargs
) -> Union[np.ndarray, NetworkEstimationResult]:
    """
    General function to estimate networks.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data or correlation matrix
    n : int, optional
        Sample size
    corr : str
        Correlation method
    na_data : str
        Missing data handling
    model : str
        Network estimation model ("glasso" or "tmfg")
    network_only : bool
        Return only the network matrix
    verbose : bool
        Print progress

    Returns
    -------
    np.ndarray or NetworkEstimationResult
        Estimated network or full results
    """
    model = model.lower()

    if model == "glasso":
        network = glasso(
            data, n, corr, na_data, verbose=verbose, **kwargs
        )
        methods = {
            "model": "glasso",
            "corr": corr,
            "na.data": na_data,
            **{k: v for k, v in kwargs.items() if k in ["gamma", "model_selection"]}
        }
    elif model == "tmfg":
        network = tmfg(
            data, n, corr, na_data, verbose=verbose, **kwargs
        )
        methods = {
            "model": "tmfg",
            "corr": corr,
            "na.data": na_data,
            "partial": kwargs.get("partial", False)
        }
    else:
        raise ValueError(f"Unknown model: {model}. Use 'glasso' or 'tmfg'")

    if network_only:
        return network

    output = obtain_sample_correlations(data, n, corr, na_data, verbose=verbose)

    return NetworkEstimationResult(
        network=network,
        methods=methods,
        correlation=output["correlation_matrix"],
        n=output["n"]
    )
