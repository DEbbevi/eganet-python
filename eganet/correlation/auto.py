"""
Automatic correlation selection.

Automatically selects appropriate correlation method based on data type:
- Pearson for continuous
- Polychoric for ordinal
- Tetrachoric for binary
- Polyserial/biserial for mixed types
"""

from __future__ import annotations
from typing import Union, Optional, Literal, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from eganet.correlation.polychoric import polychoric_matrix, polychoric_correlation_pair


CorrelationMethod = Literal["auto", "pearson", "spearman", "polychoric", "cosine"]
NAMethod = Literal["pairwise", "listwise"]


def data_categories(data: np.ndarray) -> np.ndarray:
    """
    Count unique categories for each variable.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix

    Returns
    -------
    np.ndarray
        Number of unique values per column
    """
    categories = np.zeros(data.shape[1], dtype=int)
    for i in range(data.shape[1]):
        col = data[:, i]
        valid = col[~np.isnan(col)]
        categories[i] = len(np.unique(valid))
    return categories


def detect_variable_types(
    data: np.ndarray,
    ordinal_categories: int = 7
) -> np.ndarray:
    """
    Detect variable types (binary, ordinal, continuous).

    Parameters
    ----------
    data : np.ndarray
        Input data
    ordinal_categories : int
        Maximum categories for ordinal

    Returns
    -------
    np.ndarray
        Array of type codes: 0=binary, 1=ordinal, 2=continuous
    """
    categories = data_categories(data)
    types = np.zeros(len(categories), dtype=int)

    for i, n_cat in enumerate(categories):
        if n_cat <= 2:
            types[i] = 0  # binary
        elif n_cat <= ordinal_categories:
            types[i] = 1  # ordinal
        else:
            types[i] = 2  # continuous

    return types


def pearson_correlation(
    data: np.ndarray,
    na_method: NAMethod = "pairwise"
) -> np.ndarray:
    """
    Compute Pearson correlation matrix.

    Parameters
    ----------
    data : np.ndarray
        Input data
    na_method : str
        How to handle missing data

    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    if na_method == "listwise":
        mask = ~np.any(np.isnan(data), axis=1)
        data = data[mask]

    n_vars = data.shape[1]
    corr = np.eye(n_vars)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            x, y = data[:, i], data[:, j]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                r = np.corrcoef(x[mask], y[mask])[0, 1]
            else:
                r = np.nan
            corr[i, j] = corr[j, i] = r

    return corr


def spearman_correlation(
    data: np.ndarray,
    na_method: NAMethod = "pairwise"
) -> np.ndarray:
    """
    Compute Spearman rank correlation matrix.

    Parameters
    ----------
    data : np.ndarray
        Input data
    na_method : str
        How to handle missing data

    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    if na_method == "listwise":
        mask = ~np.any(np.isnan(data), axis=1)
        data = data[mask]

    n_vars = data.shape[1]
    corr = np.eye(n_vars)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            x, y = data[:, i], data[:, j]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                r, _ = stats.spearmanr(x[mask], y[mask])
            else:
                r = np.nan
            corr[i, j] = corr[j, i] = r

    return corr


def polyserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute polyserial correlation (continuous-ordinal).

    Uses the approximation: polyserial ≈ pearson * sqrt(6) / pi
    for large samples, or polychoric with discretization.

    Parameters
    ----------
    x : np.ndarray
        Continuous variable
    y : np.ndarray
        Ordinal variable

    Returns
    -------
    float
        Polyserial correlation
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_valid, y_valid = x[mask], y[mask]

    if len(x_valid) < 3:
        return np.nan

    pearson_r = np.corrcoef(x_valid, y_valid)[0, 1]

    correction = np.sqrt(6) / np.pi
    return np.clip(pearson_r * correction, -1, 1)


def biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute biserial correlation (continuous-binary).

    Parameters
    ----------
    x : np.ndarray
        Continuous variable
    y : np.ndarray
        Binary variable (0/1)

    Returns
    -------
    float
        Biserial correlation
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_valid, y_valid = x[mask], y[mask].astype(int)

    if len(x_valid) < 3:
        return np.nan

    y_unique = np.unique(y_valid)
    if len(y_unique) != 2:
        return np.nan

    y_binary = (y_valid == y_unique[1]).astype(int)
    n = len(y_binary)
    n1 = y_binary.sum()
    n0 = n - n1

    if n1 == 0 or n0 == 0:
        return np.nan

    p = n1 / n
    q = n0 / n

    mean1 = x_valid[y_binary == 1].mean()
    mean0 = x_valid[y_binary == 0].mean()
    std_x = x_valid.std()

    if std_x == 0:
        return np.nan

    y_ordinate = stats.norm.pdf(stats.norm.ppf(p))

    r_bis = (mean1 - mean0) * (p * q) / (std_x * y_ordinate)
    return np.clip(r_bis, -1, 1)


def auto_correlate(
    data: Union[np.ndarray, pd.DataFrame],
    corr: CorrelationMethod = "auto",
    na_data: NAMethod = "pairwise",
    ordinal_categories: int = 7,
    verbose: bool = False
) -> Tuple[np.ndarray, str]:
    """
    Automatically compute appropriate correlations.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data
    corr : str
        Correlation method ("auto", "pearson", "spearman", "polychoric")
    na_data : str
        Missing data handling ("pairwise", "listwise")
    ordinal_categories : int
        Max categories for ordinal
    verbose : bool
        Print information

    Returns
    -------
    tuple
        (correlation_matrix, method_used)
    """
    if isinstance(data, pd.DataFrame):
        data = data.values.astype(float)
    else:
        data = data.astype(float)

    if corr == "pearson":
        return pearson_correlation(data, na_data), "pearson"

    if corr == "spearman":
        return spearman_correlation(data, na_data), "spearman"

    if corr == "polychoric":
        return polychoric_matrix(data, ordinal_categories, verbose=verbose), "polychoric"

    if corr == "cosine":
        from eganet.correlation.cosine import cosine_similarity
        return cosine_similarity(data), "cosine"

    var_types = detect_variable_types(data, ordinal_categories)

    all_continuous = np.all(var_types == 2)
    all_ordinal = np.all(var_types <= 1)
    mixed = not (all_continuous or all_ordinal)

    if all_continuous:
        if verbose:
            print("Using Pearson correlation (all continuous)")
        return pearson_correlation(data, na_data), "pearson"

    if all_ordinal:
        if verbose:
            print("Using polychoric correlation (all ordinal)")
        return polychoric_matrix(data, ordinal_categories, verbose=verbose), "polychoric"

    if verbose:
        print("Using mixed correlation methods")

    n_vars = data.shape[1]
    corr_matrix = np.eye(n_vars)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            type_i, type_j = var_types[i], var_types[j]
            x, y = data[:, i], data[:, j]

            if type_i == 2 and type_j == 2:
                mask = ~(np.isnan(x) | np.isnan(y))
                r = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() > 2 else 0

            elif type_i <= 1 and type_j <= 1:
                r, _ = polychoric_correlation_pair(x, y)

            elif type_i == 2 and type_j <= 1:
                if type_j == 0:
                    r = biserial_correlation(x, y)
                else:
                    r = polyserial_correlation(x, y)

            elif type_i <= 1 and type_j == 2:
                if type_i == 0:
                    r = biserial_correlation(y, x)
                else:
                    r = polyserial_correlation(y, x)
            else:
                r = 0

            if np.isnan(r):
                r = 0

            corr_matrix[i, j] = corr_matrix[j, i] = r

    return corr_matrix, "auto"


def obtain_sample_correlations(
    data: Union[np.ndarray, pd.DataFrame],
    n: Optional[int] = None,
    corr: CorrelationMethod = "auto",
    na_data: NAMethod = "pairwise",
    ordinal_categories: int = 7,
    verbose: bool = False,
    **kwargs
) -> dict:
    """
    Obtain sample correlations and sample size.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data or correlation matrix
    n : int, optional
        Sample size (required if data is correlation matrix)
    corr : str
        Correlation method
    na_data : str
        Missing data handling
    ordinal_categories : int
        Max categories for ordinal
    verbose : bool
        Print information

    Returns
    -------
    dict
        Contains 'correlation_matrix' and 'n'
    """
    if isinstance(data, pd.DataFrame):
        data_arr = data.values.astype(float)
    else:
        data_arr = data.astype(float)

    from eganet.utils.helpers import is_symmetric, is_correlation_matrix

    if is_symmetric(data_arr) and is_correlation_matrix(data_arr):
        if n is None:
            raise ValueError("Sample size 'n' required when providing correlation matrix")
        return {"correlation_matrix": data_arr, "n": n}

    if na_data == "listwise":
        mask = ~np.any(np.isnan(data_arr), axis=1)
        data_arr = data_arr[mask]

    sample_size = data_arr.shape[0]

    correlation_matrix, method = auto_correlate(
        data_arr, corr, na_data, ordinal_categories, verbose
    )

    return {"correlation_matrix": correlation_matrix, "n": sample_size, "method": method}
