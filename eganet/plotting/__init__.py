"""Plotting functions."""

from eganet.plotting.network import plot_network
from eganet.plotting.colors import color_palette_ega
from eganet.plotting.stability import (
    plot_dimension_frequency,
    plot_item_stability_heatmap,
    plot_replication_matrix,
    plot_stability_summary,
)
from eganet.plotting.comparison import (
    compare_ega_plots,
    plot_boot_comparison,
    plot_structure_comparison,
    plot_tefi_comparison,
    plot_fit_comparison,
)

plot_stability = plot_stability_summary
plot_dimension_stability = plot_dimension_frequency
plot_item_stability = plot_item_stability_heatmap

__all__ = [
    "plot_network",
    "color_palette_ega",
    "plot_stability",
    "plot_dimension_frequency",
    "plot_dimension_stability",
    "plot_item_stability",
    "plot_item_stability_heatmap",
    "plot_replication_matrix",
    "plot_stability_summary",
    "compare_ega_plots",
    "plot_boot_comparison",
    "plot_structure_comparison",
    "plot_tefi_comparison",
    "plot_fit_comparison",
]
