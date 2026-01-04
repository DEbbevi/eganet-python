"""
Network regularization with various penalty functions.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, Literal
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.covariance import graphical_lasso


PenaltyType = Literal["l1", "l2", "scad", "mcp", "atan", "bridge", "lomax"]
ICType = Literal["aic", "aicc", "bic", "ebic"]


def network_regularization(
    data: np.ndarray,
    n: Optional[int] = None,
    corr: str = "auto",
    penalty: PenaltyType = "l1",
    gamma: Optional[float] = None,
    nlambda: int = 50,
    lambda_min_ratio: float = 0.01,
    penalize_diagonal: bool = False,
    optimize_lambda: bool = False,
    ic: ICType = "bic",
    ebic_gamma: float = 0.5,
) -> Dict[str, Any]:
    """
    Estimate regularized networks with various penalties.

    Parameters
    ----------
    data : np.ndarray
        Data matrix or correlation matrix
    n : int, optional
        Sample size (required if data is correlation)
    corr : str
        Correlation method
    penalty : str
        Penalty type: "l1", "l2", "scad", "mcp", "atan", "bridge", "lomax"
    gamma : float, optional
        Shape parameter for penalty (defaults based on penalty type)
    nlambda : int
        Number of lambda values to test
    lambda_min_ratio : float
        Ratio of min to max lambda
    penalize_diagonal : bool
        Whether to penalize diagonal
    optimize_lambda : bool
        Whether to optimize lambda
    ic : str
        Information criterion: "aic", "aicc", "bic", "ebic"
    ebic_gamma : float
        EBIC gamma parameter

    Returns
    -------
    dict
        Network estimation results
    """
    from eganet.correlation.auto import auto_correlate

    if _is_correlation_matrix(data):
        S = data
        if n is None:
            raise ValueError("Sample size n required for correlation input")
    else:
        S, _ = auto_correlate(data, corr_method=corr)
        n = data.shape[0]

    nodes = S.shape[0]
    K = np.linalg.inv(S)

    if gamma is None:
        gamma = _get_default_gamma(penalty)

    derivative_fn = _get_derivative_function(penalty)

    S_zero_diag = S - np.eye(nodes)
    lambda_max = np.max(np.abs(S_zero_diag))
    lambda_min = lambda_min_ratio * lambda_max
    lambdas = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), nlambda))

    if optimize_lambda:
        result = minimize_scalar(
            lambda lam: _lambda_objective(
                lam, gamma, K, S, derivative_fn, penalize_diagonal,
                ic, n, nodes, ebic_gamma
            ),
            bounds=(0, 1 if penalty != "l2" else 10),
            method='bounded'
        )
        optimal_lambda = result.x
        optimal_ic = result.fun

        lambda_matrix = np.abs(derivative_fn(K, optimal_lambda, gamma))
        if not penalize_diagonal:
            np.fill_diagonal(lambda_matrix, 0)

        network, precision = _estimate_glasso(S, lambda_matrix)

        return {
            "network": network,
            "precision": precision,
            "correlation": S,
            "penalty": penalty,
            "lambda": optimal_lambda,
            "gamma": gamma,
            "criterion": ic,
            "ic_value": optimal_ic,
        }

    best_ic = np.inf
    best_idx = 0
    networks = []
    precisions = []

    for i, lam in enumerate(lambdas):
        lambda_matrix = np.abs(derivative_fn(K, lam, gamma))
        if not penalize_diagonal:
            np.fill_diagonal(lambda_matrix, 0)

        try:
            network, precision = _estimate_glasso(S, lambda_matrix)
            networks.append(network)
            precisions.append(precision)

            ic_value = _compute_ic(S, precision, n, nodes, ic, ebic_gamma)
            if ic_value < best_ic:
                best_ic = ic_value
                best_idx = len(networks) - 1
        except Exception:
            networks.append(None)
            precisions.append(None)

    return {
        "network": networks[best_idx],
        "precision": precisions[best_idx],
        "correlation": S,
        "penalty": penalty,
        "lambda": lambdas[best_idx],
        "gamma": gamma,
        "criterion": ic,
        "ic_value": best_ic,
    }


def _is_correlation_matrix(data: np.ndarray) -> bool:
    """Check if data is a correlation matrix."""
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        return False
    diag_ones = np.allclose(np.diag(data), 1.0)
    symmetric = np.allclose(data, data.T)
    return diag_ones and symmetric


def _get_default_gamma(penalty: str) -> float:
    """Get default gamma for penalty type."""
    defaults = {
        "atan": 0.01,
        "bridge": 1.0,
        "lomax": 4.0,
        "mcp": 3.0,
        "scad": 3.7,
        "l1": 0.0,
        "l2": 0.0,
    }
    return defaults.get(penalty, 0.0)


def _get_derivative_function(penalty: str):
    """Get derivative function for penalty type."""
    derivatives = {
        "l1": _l1_derivative,
        "l2": _l2_derivative,
        "scad": _scad_derivative,
        "mcp": _mcp_derivative,
        "atan": _atan_derivative,
        "bridge": _bridge_derivative,
        "lomax": _lomax_derivative,
    }
    return derivatives.get(penalty, _l1_derivative)


def _l1_derivative(x: np.ndarray, lam: float, gamma: float) -> np.ndarray:
    """L1 (LASSO) penalty derivative."""
    return np.full_like(x, lam)


def _l2_derivative(x: np.ndarray, lam: float, gamma: float) -> np.ndarray:
    """L2 (Ridge) penalty derivative."""
    return 2 * lam * np.abs(x)


def _scad_derivative(x: np.ndarray, lam: float, gamma: float) -> np.ndarray:
    """SCAD penalty derivative."""
    abs_x = np.abs(x)
    result = np.zeros_like(x)

    mask1 = abs_x <= lam
    result[mask1] = lam

    mask2 = (abs_x > lam) & (abs_x <= gamma * lam)
    result[mask2] = (gamma * lam - abs_x[mask2]) / (gamma - 1)

    return result


def _mcp_derivative(x: np.ndarray, lam: float, gamma: float) -> np.ndarray:
    """MCP penalty derivative."""
    abs_x = np.abs(x)
    result = np.zeros_like(x)

    mask = abs_x <= gamma * lam
    result[mask] = lam - abs_x[mask] / gamma

    return result


def _atan_derivative(x: np.ndarray, lam: float, gamma: float) -> np.ndarray:
    """Arctangent penalty derivative."""
    return lam * (gamma + 2 * np.pi) / (gamma**2 + x**2) * gamma


def _bridge_derivative(x: np.ndarray, lam: float, gamma: float) -> np.ndarray:
    """Bridge penalty derivative."""
    abs_x = np.abs(x)
    abs_x = np.maximum(abs_x, 1e-10)
    return lam * gamma * abs_x**(gamma - 1)


def _lomax_derivative(x: np.ndarray, lam: float, gamma: float) -> np.ndarray:
    """Lomax penalty derivative."""
    abs_x = np.abs(x)
    return lam * gamma / (abs_x + 1)**(gamma + 1)


def _estimate_glasso(S: np.ndarray, lambda_matrix: np.ndarray):
    """Estimate GLASSO with penalty matrix."""
    alpha = np.mean(lambda_matrix[np.triu_indices_from(lambda_matrix, k=1)])
    if alpha <= 0:
        alpha = 0.01

    try:
        precision, _ = graphical_lasso(S, alpha=alpha, max_iter=200)
        network = _precision_to_partial(precision)
        return network, precision
    except Exception:
        return np.zeros_like(S), np.eye(S.shape[0])


def _precision_to_partial(K: np.ndarray) -> np.ndarray:
    """Convert precision matrix to partial correlation network."""
    diag_sqrt = np.sqrt(np.diag(K))
    diag_sqrt = np.maximum(diag_sqrt, 1e-10)
    D = np.diag(1.0 / diag_sqrt)
    P = -D @ K @ D
    np.fill_diagonal(P, 0)
    return P


def _compute_ic(
    S: np.ndarray,
    K: np.ndarray,
    n: int,
    nodes: int,
    ic: str,
    ebic_gamma: float
) -> float:
    """Compute information criterion."""
    try:
        det_K = np.linalg.det(K)
        if det_K <= 0:
            return np.inf

        L = -2 * (n / 2) * (np.log(det_K) - np.trace(S @ K))
        E = np.sum(np.abs(K[np.triu_indices_from(K, k=1)]) > 1e-10)

        if ic == "aic":
            return L + 2 * E
        elif ic == "aicc":
            if n - E - 1 <= 0:
                return np.inf
            return L + 2 * E + (2 * E**2 + 2 * E) / (n - E - 1)
        elif ic == "bic":
            return L + E * np.log(n)
        elif ic == "ebic":
            return L + E * np.log(n) + 4 * E * ebic_gamma * np.log(nodes)
        else:
            return L + E * np.log(n)
    except Exception:
        return np.inf


def _lambda_objective(
    lam: float,
    gamma: float,
    K: np.ndarray,
    S: np.ndarray,
    derivative_fn,
    penalize_diagonal: bool,
    ic: str,
    n: int,
    nodes: int,
    ebic_gamma: float
) -> float:
    """Objective function for lambda optimization."""
    lambda_matrix = np.abs(derivative_fn(K, lam, gamma))
    if not penalize_diagonal:
        np.fill_diagonal(lambda_matrix, 0)

    try:
        network, precision = _estimate_glasso(S, lambda_matrix)
        ic_value = _compute_ic(S, precision, n, nodes, ic, ebic_gamma)
        return ic_value if np.isfinite(ic_value) else np.inf
    except Exception:
        return np.inf
