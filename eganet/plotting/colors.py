"""
Color palettes for EGA visualization.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np


def color_palette_ega(
    n_colors: int = 8,
    palette: str = "default"
) -> List[str]:
    """
    Get EGAnet color palette.

    Parameters
    ----------
    n_colors : int
        Number of colors needed
    palette : str
        Palette name ("default", "colorblind", "pastel")

    Returns
    -------
    list
        List of hex color codes
    """
    palettes = {
        "default": [
            "#E41A1C",  # Red
            "#377EB8",  # Blue
            "#4DAF4A",  # Green
            "#984EA3",  # Purple
            "#FF7F00",  # Orange
            "#FFFF33",  # Yellow
            "#A65628",  # Brown
            "#F781BF",  # Pink
            "#999999",  # Gray
            "#66C2A5",  # Teal
            "#FC8D62",  # Salmon
            "#8DA0CB",  # Light blue
        ],
        "colorblind": [
            "#0072B2",  # Blue
            "#E69F00",  # Orange
            "#009E73",  # Green
            "#CC79A7",  # Pink
            "#F0E442",  # Yellow
            "#56B4E9",  # Sky blue
            "#D55E00",  # Vermillion
            "#000000",  # Black
        ],
        "pastel": [
            "#8DD3C7",
            "#FFFFB3",
            "#BEBADA",
            "#FB8072",
            "#80B1D3",
            "#FDB462",
            "#B3DE69",
            "#FCCDE5",
            "#D9D9D9",
            "#BC80BD",
        ],
    }

    base_palette = palettes.get(palette, palettes["default"])

    if n_colors <= len(base_palette):
        return base_palette[:n_colors]

    result = base_palette.copy()
    while len(result) < n_colors:
        idx = len(result) % len(base_palette)
        lightened = _lighten_color(base_palette[idx], 0.3)
        result.append(lightened)

    return result[:n_colors]


def _lighten_color(hex_color: str, amount: float = 0.3) -> str:
    """Lighten a hex color."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    lightened = tuple(int(c + (255 - c) * amount) for c in rgb)

    return "#{:02x}{:02x}{:02x}".format(*lightened)


def _darken_color(hex_color: str, amount: float = 0.3) -> str:
    """Darken a hex color."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    darkened = tuple(int(c * (1 - amount)) for c in rgb)

    return "#{:02x}{:02x}{:02x}".format(*darkened)


def edge_color_palette() -> dict:
    """
    Get default edge colors.

    Returns
    -------
    dict
        Dictionary with positive and negative edge colors
    """
    return {
        "positive": "#377EB8",
        "negative": "#E41A1C",
    }


def continuous_colormap(
    n_colors: int = 256,
    cmap_name: str = "RdBu_r"
):
    """
    Get continuous colormap for correlation-like values.

    Parameters
    ----------
    n_colors : int
        Number of discrete colors
    cmap_name : str
        Matplotlib colormap name

    Returns
    -------
    colormap
        Matplotlib colormap
    """
    import matplotlib.pyplot as plt
    return plt.cm.get_cmap(cmap_name, n_colors)
