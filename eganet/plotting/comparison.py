"""
Comparison plots for EGA results.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from eganet.utils.helpers import EGAResult, BootEGAResult


def compare_ega_plots(
    ega_results: List["EGAResult"],
    names: Optional[List[str]] = None,
    figsize: tuple = (15, 5),
    layout: Optional[tuple] = None,
    **kwargs
) -> Any:
    """
    Compare multiple EGA results side by side.

    Parameters
    ----------
    ega_results : list
        List of EGAResult objects to compare
    names : list, optional
        Names for each result
    figsize : tuple
        Figure size
    layout : tuple, optional
        Subplot layout (rows, cols)
    **kwargs
        Additional arguments passed to plot_network

    Returns
    -------
    matplotlib.figure.Figure
        Comparison figure
    """
    import matplotlib.pyplot as plt
    from eganet.plotting.network import plot_network

    n_results = len(ega_results)

    if names is None:
        names = [f"EGA {i+1}" for i in range(n_results)]

    if layout is None:
        n_cols = min(3, n_results)
        n_rows = (n_results + n_cols - 1) // n_cols
        layout = (n_rows, n_cols)

    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)

    if n_results == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (ega, name) in enumerate(zip(ega_results, names)):
        if i < len(axes):
            plt.sca(axes[i])
            plot_network(ega, title=name, ax=axes[i], **kwargs)

    for i in range(n_results, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def plot_boot_comparison(
    boot_result: "BootEGAResult",
    figsize: tuple = (12, 5),
    **kwargs
) -> Any:
    """
    Plot bootstrap EGA comparison.

    Parameters
    ----------
    boot_result : BootEGAResult
        Bootstrap EGA result
    figsize : tuple
        Figure size
    **kwargs
        Additional plotting arguments

    Returns
    -------
    matplotlib.figure.Figure
        Bootstrap comparison figure
    """
    import matplotlib.pyplot as plt
    from eganet.plotting.network import plot_network

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plt.sca(axes[0])
    plot_network(boot_result.ega, title="Typical Network", ax=axes[0], **kwargs)

    dim_freqs = {}
    for result in boot_result.boot_results:
        dim = result.n_dim
        dim_freqs[dim] = dim_freqs.get(dim, 0) + 1

    dims = sorted(dim_freqs.keys())
    freqs = [dim_freqs[d] for d in dims]
    total = sum(freqs)
    percentages = [f / total * 100 for f in freqs]

    axes[1].bar(dims, percentages, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Number of Dimensions')
    axes[1].set_ylabel('Frequency (%)')
    axes[1].set_title('Bootstrap Dimension Distribution')
    axes[1].set_xticks(dims)

    for i, (d, p) in enumerate(zip(dims, percentages)):
        axes[1].annotate(
            f'{p:.1f}%',
            xy=(d, p),
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    return fig


def plot_structure_comparison(
    ega_result: "EGAResult",
    alternative_wc: np.ndarray,
    names: tuple = ("EGA", "Alternative"),
    figsize: tuple = (12, 5),
    **kwargs
) -> Any:
    """
    Compare EGA structure with alternative structure.

    Parameters
    ----------
    ega_result : EGAResult
        Original EGA result
    alternative_wc : np.ndarray
        Alternative community membership
    names : tuple
        Names for the two structures
    figsize : tuple
        Figure size
    **kwargs
        Additional plotting arguments

    Returns
    -------
    matplotlib.figure.Figure
        Structure comparison figure
    """
    import matplotlib.pyplot as plt
    from eganet.plotting.network import plot_network
    from eganet.utils.helpers import EGAResult

    alt_result = EGAResult(
        network=ega_result.network,
        wc=alternative_wc,
        n_dim=len(np.unique(alternative_wc[~np.isnan(alternative_wc)])),
        correlation=ega_result.correlation,
        n=ega_result.n,
        dim_variables=None,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plt.sca(axes[0])
    plot_network(ega_result, title=names[0], ax=axes[0], **kwargs)

    plt.sca(axes[1])
    plot_network(alt_result, title=names[1], ax=axes[1], **kwargs)

    plt.tight_layout()
    return fig


def plot_tefi_comparison(
    results: Dict[str, float],
    figsize: tuple = (8, 5),
) -> Any:
    """
    Compare TEFI values across different structures/methods.

    Parameters
    ----------
    results : dict
        Dictionary of {name: tefi_value}
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        TEFI comparison bar plot
    """
    import matplotlib.pyplot as plt

    names = list(results.keys())
    values = list(results.values())

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['green' if v == min(values) else 'steelblue' for v in values]

    bars = ax.bar(names, values, color=colors, edgecolor='black')

    ax.set_ylabel('TEFI')
    ax.set_title('TEFI Comparison (lower is better)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, values):
        ax.annotate(
            f'{val:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, val),
            ha='center',
            va='bottom' if val >= 0 else 'top',
            fontsize=9
        )

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_fit_comparison(
    fit_results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    figsize: tuple = (10, 6),
) -> Any:
    """
    Compare fit indices across different networks/methods.

    Parameters
    ----------
    fit_results : dict
        Dictionary of {name: {metric: value}}
    metrics : list, optional
        Metrics to plot (defaults to RMSEA, CFI, SRMR)
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Fit comparison plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if metrics is None:
        metrics = ["rmsea", "cfi", "srmr"]

    names = list(fit_results.keys())
    n_metrics = len(metrics)
    n_methods = len(names)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_methods)
    width = 0.6

    for i, metric in enumerate(metrics):
        values = [fit_results[name].get(metric, np.nan) for name in names]

        axes[i].bar(x, values, width, color='steelblue', edgecolor='black')
        axes[i].set_ylabel(metric.upper())
        axes[i].set_title(metric.upper())
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(names, rotation=45, ha='right')

        if metric == "rmsea":
            axes[i].axhline(y=0.08, color='red', linestyle='--', alpha=0.7, label='Cutoff')
            axes[i].axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Good')
        elif metric == "cfi":
            axes[i].axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='Cutoff')
            axes[i].axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='Good')
        elif metric == "srmr":
            axes[i].axhline(y=0.08, color='red', linestyle='--', alpha=0.7, label='Cutoff')

    plt.tight_layout()
    return fig


def plot_dimension_agreement(
    wc1: np.ndarray,
    wc2: np.ndarray,
    names: tuple = ("Method 1", "Method 2"),
    figsize: tuple = (8, 6),
) -> Any:
    """
    Plot agreement between two community structures.

    Parameters
    ----------
    wc1 : np.ndarray
        First community membership vector
    wc2 : np.ndarray
        Second community membership vector
    names : tuple
        Names for the two methods
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Agreement heatmap
    """
    import matplotlib.pyplot as plt

    mask = ~np.isnan(wc1) & ~np.isnan(wc2)
    wc1_clean = wc1[mask].astype(int)
    wc2_clean = wc2[mask].astype(int)

    comm1 = np.unique(wc1_clean)
    comm2 = np.unique(wc2_clean)

    contingency = np.zeros((len(comm1), len(comm2)))
    for i, c1 in enumerate(comm1):
        for j, c2 in enumerate(comm2):
            contingency[i, j] = np.sum((wc1_clean == c1) & (wc2_clean == c2))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(contingency, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='Count')

    ax.set_xticks(range(len(comm2)))
    ax.set_xticklabels([f'C{c}' for c in comm2])
    ax.set_yticks(range(len(comm1)))
    ax.set_yticklabels([f'C{c}' for c in comm1])

    ax.set_xlabel(names[1])
    ax.set_ylabel(names[0])
    ax.set_title('Community Agreement')

    for i in range(len(comm1)):
        for j in range(len(comm2)):
            ax.annotate(
                f'{int(contingency[i, j])}',
                xy=(j, i),
                ha='center',
                va='center',
                color='white' if contingency[i, j] > contingency.max() / 2 else 'black'
            )

    plt.tight_layout()
    return fig
