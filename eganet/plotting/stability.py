"""
Stability visualization for bootstrap EGA.
"""

from __future__ import annotations
from typing import Union, Optional, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from eganet.utils.helpers import BootEGAResult


def plot_dimension_frequency(
    boot_result: "BootEGAResult",
    ax=None,
    color: str = "steelblue",
    **kwargs
):
    """
    Plot frequency distribution of number of dimensions.

    Parameters
    ----------
    boot_result : BootEGAResult
        Bootstrap EGA result
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    color : str
        Bar color

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    freq = boot_result.frequency
    dims = np.arange(len(freq))
    non_zero = freq > 0

    ax.bar(dims[non_zero], freq[non_zero], color=color, alpha=0.7, edgecolor="black")

    ax.set_xlabel("Number of Dimensions", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Bootstrap Dimension Stability", fontsize=14)

    modal_dim = np.argmax(freq)
    ax.axvline(modal_dim, color="red", linestyle="--", linewidth=2,
               label=f"Modal: {modal_dim}")
    ax.legend()

    return fig


def plot_item_stability_heatmap(
    boot_result: "BootEGAResult",
    ax=None,
    cmap: str = "RdYlGn",
    **kwargs
):
    """
    Plot item stability as heatmap.

    Parameters
    ----------
    boot_result : BootEGAResult
        Bootstrap EGA result
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    cmap : str
        Colormap name

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    items = list(boot_result.item_stability.keys())
    stabs = np.array(list(boot_result.item_stability.values())).reshape(-1, 1)

    im = ax.imshow(stabs, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(items)
    ax.set_xticks([])

    cbar = plt.colorbar(im, ax=ax, label="Stability")
    ax.set_title("Item Stability", fontsize=14)

    return fig


def plot_replication_matrix(
    replication: np.ndarray,
    var_names: Optional[List[str]] = None,
    ax=None,
    cmap: str = "Blues",
    **kwargs
):
    """
    Plot co-assignment replication matrix.

    Parameters
    ----------
    replication : np.ndarray
        Replication matrix
    var_names : list, optional
        Variable names
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    cmap : str
        Colormap name

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    n = replication.shape[0]

    if var_names is None:
        var_names = [f"V{i+1}" for i in range(n)]

    im = ax.imshow(replication, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(var_names, rotation=90)
    ax.set_yticklabels(var_names)

    plt.colorbar(im, ax=ax, label="Replication Proportion")
    ax.set_title("Co-assignment Replication Matrix", fontsize=14)

    return fig


def plot_stability_summary(
    boot_result: "BootEGAResult",
    figsize: tuple = (12, 8),
    **kwargs
):
    """
    Create summary plot of stability metrics.

    Parameters
    ----------
    boot_result : BootEGAResult
        Bootstrap EGA result
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    from eganet.psychometrics.stability import replication_matrix

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    plot_dimension_frequency(boot_result, ax=axes[0, 0])

    items = list(boot_result.item_stability.keys())
    stabs = list(boot_result.item_stability.values())
    y_pos = np.arange(len(items))
    axes[0, 1].barh(y_pos, stabs, color="steelblue", alpha=0.7)
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(items, fontsize=8)
    axes[0, 1].set_xlabel("Stability")
    axes[0, 1].set_title("Item Stability")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].axvline(0.75, color="red", linestyle="--", alpha=0.5)

    try:
        rep_matrix = replication_matrix(boot_result)
        if rep_matrix.size > 0:
            im = axes[1, 0].imshow(rep_matrix, cmap="Blues", vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[1, 0])
            axes[1, 0].set_title("Replication Matrix")
    except Exception:
        axes[1, 0].text(0.5, 0.5, "Replication matrix\nnot available",
                        ha="center", va="center")
        axes[1, 0].set_title("Replication Matrix")

    summary_text = [
        f"Total bootstraps: {boot_result.methods.get('n_boots', 'N/A')}",
        f"Successful: {boot_result.methods.get('successful_boots', 'N/A')}",
        f"Modal dimensions: {np.argmax(boot_result.frequency)}",
        f"Mean item stability: {np.mean(stabs):.3f}",
    ]
    axes[1, 1].text(0.1, 0.5, "\n".join(summary_text), fontsize=12,
                    transform=axes[1, 1].transAxes, verticalalignment="center")
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Summary Statistics")

    plt.tight_layout()
    return fig
