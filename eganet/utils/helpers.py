"""
Core utility functions for EGAnet.

Provides matrix operations, validation, type checking, and helper functions
used throughout the package.
"""

from __future__ import annotations
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class EGAResult:
    """Result container for EGA analysis."""
    network: np.ndarray
    wc: np.ndarray
    n_dim: int
    correlation: np.ndarray
    n: int
    dim_variables: pd.DataFrame
    tefi: Optional[float] = None
    methods: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        model = self.methods.get("model", "unknown")
        corr = self.methods.get("corr", "unknown")
        algorithm = self.methods.get("algorithm", "unknown")

        lines = [
            f"Model: {model.upper()}",
            f"Correlations: {corr}",
            f"Algorithm: {algorithm}",
            "",
            f"Number of nodes: {self.network.shape[0]}",
            f"Number of dimensions: {self.n_dim}",
        ]

        if self.tefi is not None:
            lines.append(f"TEFI: {self.tefi:.3f}")

        return "\n".join(lines)


@dataclass
class BootEGAResult:
    """Result container for bootstrap EGA analysis."""
    ega: EGAResult
    boot_results: List[EGAResult]
    frequency: np.ndarray
    typical_network: np.ndarray
    item_stability: Dict[str, float]
    dimension_stability: Dict[int, float]
    methods: Dict[str, Any] = field(default_factory=dict)


def ensure_dimension_names(
    data: Union[np.ndarray, pd.DataFrame]
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Ensure data has dimension names (row and column names for matrices).

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data matrix

    Returns
    -------
    np.ndarray or pd.DataFrame
        Data with names assigned if missing
    """
    if isinstance(data, pd.DataFrame):
        if data.columns.tolist() == list(range(len(data.columns))):
            data.columns = [f"V{i+1}" for i in range(len(data.columns))]
        return data

    return data


def get_variable_names(data: Union[np.ndarray, pd.DataFrame]) -> List[str]:
    """Get variable names from data."""
    if isinstance(data, pd.DataFrame):
        return list(data.columns)
    return [f"V{i+1}" for i in range(data.shape[1])]


def is_symmetric(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is symmetric.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix
    tol : float
        Tolerance for floating point comparison

    Returns
    -------
    bool
        True if matrix is symmetric
    """
    if matrix.ndim != 2:
        return False
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return np.allclose(matrix, matrix.T, atol=tol)


def is_correlation_matrix(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if matrix is a valid correlation matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix
    tol : float
        Tolerance for comparison

    Returns
    -------
    bool
        True if valid correlation matrix
    """
    if not is_symmetric(matrix, tol):
        return False
    diag = np.diag(matrix)
    if not np.allclose(diag, 1.0, atol=tol):
        return False
    if np.any(matrix < -1 - tol) or np.any(matrix > 1 + tol):
        return False
    return True


def transfer_names(
    source: Union[np.ndarray, pd.DataFrame],
    target: np.ndarray
) -> np.ndarray:
    """
    Transfer dimension names from source to target matrix.

    Parameters
    ----------
    source : np.ndarray or pd.DataFrame
        Source with names
    target : np.ndarray
        Target array to apply names to

    Returns
    -------
    np.ndarray
        Target array (names stored separately as we use numpy)
    """
    return target


def reindex_memberships(memberships: np.ndarray) -> np.ndarray:
    """
    Re-index community memberships to be sequential from 1.

    Parameters
    ----------
    memberships : np.ndarray
        Original community assignments (may have gaps)

    Returns
    -------
    np.ndarray
        Reindexed memberships starting from 1
    """
    result = memberships.copy().astype(float)

    valid_mask = ~np.isnan(result)
    if not np.any(valid_mask):
        return result

    unique_communities = np.unique(result[valid_mask])
    unique_communities = unique_communities[~np.isnan(unique_communities)]

    mapping = {old: new + 1 for new, old in enumerate(sorted(unique_communities))}

    for old, new in mapping.items():
        result[result == old] = new

    return result


def data_categories(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Count unique categories for each variable.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data

    Returns
    -------
    np.ndarray
        Number of unique values per column
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    categories = np.zeros(data.shape[1], dtype=int)
    for i in range(data.shape[1]):
        col = data[:, i]
        valid = col[~np.isnan(col)]
        categories[i] = len(np.unique(valid))

    return categories


def usable_data(
    data: Union[np.ndarray, pd.DataFrame],
    verbose: bool = False
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Clean data by removing unusable rows/columns.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data
    verbose : bool
        Whether to print warnings

    Returns
    -------
    np.ndarray or pd.DataFrame
        Cleaned data
    """
    if isinstance(data, pd.DataFrame):
        all_nan_cols = data.isna().all()
        if all_nan_cols.any() and verbose:
            print(f"Removing columns with all missing: {list(data.columns[all_nan_cols])}")
        data = data.loc[:, ~all_nan_cols]

        zero_var_cols = data.std() == 0
        if zero_var_cols.any() and verbose:
            print(f"Removing zero-variance columns: {list(data.columns[zero_var_cols])}")
        data = data.loc[:, ~zero_var_cols]

        return data

    all_nan_cols = np.all(np.isnan(data), axis=0)
    data = data[:, ~all_nan_cols]

    with np.errstate(all='ignore'):
        zero_var = np.nanstd(data, axis=0) == 0
    data = data[:, ~zero_var]

    return data


def lower_triangle(matrix: np.ndarray, diagonal: bool = False) -> np.ndarray:
    """
    Extract lower triangle of matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix
    diagonal : bool
        Whether to include diagonal

    Returns
    -------
    np.ndarray
        Lower triangle values as 1D array
    """
    k = 0 if diagonal else -1
    return matrix[np.tril_indices_from(matrix, k=k)]


def matrix_from_lower_triangle(
    values: np.ndarray,
    n: int,
    diagonal: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reconstruct symmetric matrix from lower triangle.

    Parameters
    ----------
    values : np.ndarray
        Lower triangle values
    n : int
        Matrix dimension
    diagonal : np.ndarray, optional
        Diagonal values (defaults to 1s for correlation matrix)

    Returns
    -------
    np.ndarray
        Symmetric matrix
    """
    matrix = np.zeros((n, n))
    matrix[np.tril_indices(n, k=-1)] = values
    matrix = matrix + matrix.T

    if diagonal is None:
        np.fill_diagonal(matrix, 1.0)
    else:
        np.fill_diagonal(matrix, diagonal)

    return matrix


def fast_table(values: np.ndarray) -> Dict[float, int]:
    """
    Fast frequency table for values.

    Parameters
    ----------
    values : np.ndarray
        Input values

    Returns
    -------
    dict
        Value -> count mapping
    """
    unique, counts = np.unique(values[~np.isnan(values)], return_counts=True)
    return dict(zip(unique, counts))


def unique_length(values: np.ndarray) -> int:
    """Count unique non-NA values."""
    valid = values[~np.isnan(values)] if np.issubdtype(values.dtype, np.floating) else values
    return len(np.unique(valid))


def format_decimal(value: float, digits: int = 3) -> str:
    """Format float with specified decimal places."""
    return f"{value:.{digits}f}"
