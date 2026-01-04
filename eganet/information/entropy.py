"""
Entropy measures for EGA.

Implements Von Neumann entropy and related entropy-based fit measures.
"""

from __future__ import annotations
from typing import Union, Dict, Any, Optional
import numpy as np
from scipy import linalg


def vn_entropy(
    data: np.ndarray,
    normalized: bool = True
) -> float:
    """
    Compute Von Neumann entropy.

    Parameters
    ----------
    data : np.ndarray
        Correlation matrix or covariance matrix
    normalized : bool
        Whether to normalize by log(n)

    Returns
    -------
    float
        Von Neumann entropy value

    Notes
    -----
    Von Neumann entropy is computed as:
    S = -sum(λ * log(λ)) for positive eigenvalues λ

    The matrix is first normalized so its trace equals 1.
    """
    eigenvalues = np.linalg.eigvalsh(data)

    eigenvalues = np.real(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    eigenvalues = eigenvalues / eigenvalues.sum()

    entropy = -np.sum(eigenvalues * np.log(eigenvalues))

    if normalized:
        n = data.shape[0]
        max_entropy = np.log(n)
        if max_entropy > 0:
            entropy = entropy / max_entropy

    return entropy


def entropy_fit(
    data: np.ndarray,
    structure: np.ndarray
) -> Dict[str, float]:
    """
    Compute entropy fit indices.

    Parameters
    ----------
    data : np.ndarray
        Correlation matrix
    structure : np.ndarray
        Community membership vector

    Returns
    -------
    dict
        Dictionary with entropy fit indices
    """
    n = data.shape[0]
    valid_structure = structure[~np.isnan(structure)]
    communities = np.unique(valid_structure)
    n_communities = len(communities)

    total_entropy = vn_entropy(data, normalized=False)

    within_entropy = 0.0
    between_entropy = 0.0

    for comm in communities:
        mask = structure == comm
        if np.sum(mask) > 1:
            submatrix = data[np.ix_(mask, mask)]
            within_entropy += vn_entropy(submatrix, normalized=False)

    if n_communities > 1:
        for i, comm_i in enumerate(communities):
            for comm_j in communities[i+1:]:
                mask_i = structure == comm_i
                mask_j = structure == comm_j
                between_matrix = data[np.ix_(mask_i, mask_j)]
                singular_values = np.linalg.svd(between_matrix, compute_uv=False)
                singular_values = singular_values[singular_values > 1e-10]
                if len(singular_values) > 0:
                    singular_values = singular_values / singular_values.sum()
                    between_entropy += -np.sum(singular_values * np.log(singular_values))

    entropy_fit_value = total_entropy - within_entropy

    return {
        "total_entropy": total_entropy,
        "within_entropy": within_entropy,
        "between_entropy": between_entropy,
        "entropy_fit": entropy_fit_value,
        "n_communities": n_communities,
    }


def shannon_entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy.

    Parameters
    ----------
    probabilities : np.ndarray
        Probability distribution (should sum to 1)

    Returns
    -------
    float
        Shannon entropy in nats
    """
    p = probabilities[probabilities > 0]
    return -np.sum(p * np.log(p))


def joint_entropy(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 10
) -> float:
    """
    Compute joint entropy between two variables.

    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable
    bins : int
        Number of bins for histogram

    Returns
    -------
    float
        Joint entropy
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[mask]
    y_valid = y[mask]

    if len(x_valid) < 2:
        return 0.0

    hist_2d, _, _ = np.histogram2d(x_valid, y_valid, bins=bins)
    joint_prob = hist_2d / hist_2d.sum()

    return shannon_entropy(joint_prob.flatten())


def conditional_entropy(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 10
) -> float:
    """
    Compute conditional entropy H(X|Y).

    Parameters
    ----------
    x : np.ndarray
        Target variable
    y : np.ndarray
        Conditioning variable
    bins : int
        Number of bins

    Returns
    -------
    float
        Conditional entropy
    """
    h_xy = joint_entropy(x, y, bins)
    h_y = shannon_entropy_variable(y, bins)

    return h_xy - h_y


def shannon_entropy_variable(x: np.ndarray, bins: int = 10) -> float:
    """Compute Shannon entropy of a single variable."""
    x_valid = x[~np.isnan(x)]
    if len(x_valid) < 2:
        return 0.0

    hist, _ = np.histogram(x_valid, bins=bins)
    prob = hist / hist.sum()

    return shannon_entropy(prob)


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 10
) -> float:
    """
    Compute mutual information between two variables.

    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable
    bins : int
        Number of bins

    Returns
    -------
    float
        Mutual information
    """
    h_x = shannon_entropy_variable(x, bins)
    h_y = shannon_entropy_variable(y, bins)
    h_xy = joint_entropy(x, y, bins)

    return h_x + h_y - h_xy


def total_correlation_entropy(
    data: np.ndarray,
    bins: int = 10
) -> float:
    """
    Compute total correlation (multi-information) from entropy.

    Parameters
    ----------
    data : np.ndarray
        Data matrix
    bins : int
        Number of bins

    Returns
    -------
    float
        Total correlation
    """
    n_vars = data.shape[1]

    marginal_sum = sum(
        shannon_entropy_variable(data[:, i], bins)
        for i in range(n_vars)
    )

    joint = _multivariate_entropy(data, bins)

    return marginal_sum - joint


def _multivariate_entropy(data: np.ndarray, bins: int = 10) -> float:
    """Compute multivariate entropy using histogram."""
    n_vars = data.shape[1]

    mask = ~np.any(np.isnan(data), axis=1)
    data_valid = data[mask]

    if len(data_valid) < 2:
        return 0.0

    hist, _ = np.histogramdd(data_valid, bins=bins)
    prob = hist / hist.sum()

    return shannon_entropy(prob.flatten())
