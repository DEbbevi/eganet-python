"""
Network visualization for EGA.
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from eganet.utils.helpers import EGAResult


def plot_network(
    ega_result: Union["EGAResult", np.ndarray],
    wc: Optional[np.ndarray] = None,
    layout: str = "spring",
    node_size: float = 300,
    edge_width_scale: float = 2.0,
    show_labels: bool = True,
    title: Optional[str] = None,
    ax=None,
    **kwargs
):
    """
    Plot network with community coloring.

    Parameters
    ----------
    ega_result : EGAResult or np.ndarray
        EGA result or network adjacency matrix
    wc : np.ndarray, optional
        Community memberships (required if network matrix provided)
    layout : str
        Layout algorithm ("spring", "circular", "kamada_kawai")
    node_size : float
        Node size
    edge_width_scale : float
        Edge width multiplier
    show_labels : bool
        Show node labels
    title : str, optional
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from eganet.plotting.colors import color_palette_ega

    if hasattr(ega_result, "network"):
        network = ega_result.network
        wc = ega_result.wc
        var_names = list(ega_result.dim_variables["items"])
    else:
        network = ega_result
        if wc is None:
            wc = np.ones(network.shape[0])
        var_names = [f"V{i+1}" for i in range(network.shape[0])]

    from eganet.utils.conversion import convert_to_networkx
    G = convert_to_networkx(network)

    mapping = {i: name for i, name in enumerate(var_names)}
    G = nx.relabel_nodes(G, mapping)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    valid_wc = wc[~np.isnan(wc)]
    communities = np.unique(valid_wc)
    n_communities = len(communities)

    colors = color_palette_ega(n_communities)
    node_colors = []

    for i, name in enumerate(var_names):
        if np.isnan(wc[i]):
            node_colors.append("gray")
        else:
            comm_idx = int(wc[i]) - 1
            node_colors.append(colors[comm_idx % len(colors)])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    edges = G.edges(data=True)
    edge_colors = []
    edge_widths = []

    for u, v, d in edges:
        weight = d.get("weight", 0)
        if weight > 0:
            edge_colors.append("blue")
        else:
            edge_colors.append("red")
        edge_widths.append(abs(weight) * edge_width_scale)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.9
    )

    if show_labels:
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=8,
            font_weight="bold"
        )

    if title:
        ax.set_title(title)
    elif hasattr(ega_result, "n_dim"):
        ax.set_title(f"EGA Network ({ega_result.n_dim} dimensions)")

    ax.axis("off")

    return fig


def plot_boot_stability(
    boot_result,
    plot_type: str = "dimension",
    ax=None,
    **kwargs
):
    """
    Plot bootstrap stability results.

    Parameters
    ----------
    boot_result : BootEGAResult
        Bootstrap EGA result
    plot_type : str
        Type of plot ("dimension", "item")
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

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

    if plot_type == "dimension":
        dims = list(boot_result.dimension_stability.keys())
        freqs = list(boot_result.dimension_stability.values())

        ax.bar(dims, freqs, color="steelblue", alpha=0.7)
        ax.set_xlabel("Number of Dimensions")
        ax.set_ylabel("Frequency")
        ax.set_title("Dimension Stability")

    else:
        items = list(boot_result.item_stability.keys())
        stabs = list(boot_result.item_stability.values())

        y_pos = np.arange(len(items))
        ax.barh(y_pos, stabs, color="steelblue", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(items)
        ax.set_xlabel("Stability")
        ax.set_title("Item Stability")
        ax.set_xlim(0, 1)

    return fig


def compare_ega_plots(
    results: List,
    names: Optional[List[str]] = None,
    layout: str = "spring",
    **kwargs
):
    """
    Compare multiple EGA solutions side by side.

    Parameters
    ----------
    results : list
        List of EGA results
    names : list, optional
        Names for each result
    layout : str
        Layout algorithm

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    n_results = len(results)

    if names is None:
        names = [f"Model {i+1}" for i in range(n_results)]

    fig, axes = plt.subplots(1, n_results, figsize=(5 * n_results, 5))

    if n_results == 1:
        axes = [axes]

    for ax, result, name in zip(axes, results, names):
        plot_network(result, layout=layout, ax=ax, title=name)

    plt.tight_layout()
    return fig
