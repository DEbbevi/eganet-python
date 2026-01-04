"""
EGAnet - Exploratory Graph Analysis for Python

A Python implementation of the EGAnet R package for estimating dimensionality
in multivariate data using network psychometrics and community detection.
"""

__version__ = "0.1.0"
__author__ = "Converted from R package by Hudson Golino, Alexander P. Christensen et al."

from eganet.core.ega import EGA, ega_estimate
from eganet.core.boot_ega import boot_ega
from eganet.core.hier_ega import hier_ega
from eganet.core.dyn_ega import dyn_ega
from eganet.core.ri_ega import ri_ega

from eganet.network.estimation import network_estimation, glasso, tmfg
from eganet.network.community import community_detection, community_consensus
from eganet.network.metrics import modularity, weighted_topological_overlap, frobenius_norm
modularity_q = modularity
wto = weighted_topological_overlap
from eganet.network.comparison import network_compare
from eganet.network.regularization import network_regularization
from eganet.network.fit import network_fit
from eganet.network.predictability import network_predictability

from eganet.correlation.auto import auto_correlate
from eganet.correlation.polychoric import polychoric_matrix
from eganet.correlation.cosine import cosine_similarity

from eganet.information.tefi import tefi, gen_tefi
from eganet.information.entropy import vn_entropy, entropy_fit
from eganet.information.jsd import jsd
from eganet.information.ergodicity import ergo_info, boot_ergo_info
from eganet.information.total_cor import total_cor
from eganet.information.clustering import info_cluster

from eganet.psychometrics.net_loads import net_loads
from eganet.psychometrics.net_scores import net_scores
from eganet.psychometrics.uva import uva
from eganet.psychometrics.stability import dimension_stability, item_stability
from eganet.psychometrics.diagnostics import item_diagnostics
from eganet.psychometrics.cfa import cfa
from eganet.psychometrics.invariance import invariance, measurement_invariance_bootstrap
from eganet.psychometrics.lct import lct

from eganet.simulation.sim_egm import sim_egm
from eganet.simulation.sim_dfm import sim_dfm
from eganet.simulation.known_graph import known_graph
from eganet.simulation.glla import glla

from eganet.plotting.network import plot_network
from eganet.plotting.colors import color_palette_ega
from eganet.plotting.stability import plot_stability_summary as plot_stability
from eganet.plotting.comparison import compare_ega_plots

from eganet.data.datasets import load_wmt2, load_depression, load_optimism, load_simulation_data

from eganet.utils.helpers import EGAResult, BootEGAResult
from eganet.utils.embedding import embed
from eganet.utils.conversion import to_igraph, from_igraph

__all__ = [
    # Core EGA
    "EGA",
    "ega_estimate",
    "boot_ega",
    "hier_ega",
    "dyn_ega",
    "ri_ega",
    # Network
    "network_estimation",
    "glasso",
    "tmfg",
    "community_detection",
    "community_consensus",
    "modularity_q",
    "wto",
    "frobenius_norm",
    "network_compare",
    "network_regularization",
    "network_fit",
    "network_predictability",
    # Correlation
    "auto_correlate",
    "polychoric_matrix",
    "cosine_similarity",
    # Information
    "tefi",
    "gen_tefi",
    "vn_entropy",
    "entropy_fit",
    "jsd",
    "ergo_info",
    "boot_ergo_info",
    "total_cor",
    "info_cluster",
    # Psychometrics
    "net_loads",
    "net_scores",
    "uva",
    "dimension_stability",
    "item_stability",
    "item_diagnostics",
    "cfa",
    "invariance",
    "lct",
    # Simulation
    "sim_egm",
    "sim_dfm",
    "known_graph",
    "glla",
    # Plotting
    "plot_network",
    "color_palette_ega",
    "plot_stability",
    "compare_ega_plots",
    # Data
    "load_wmt2",
    "load_depression",
    "load_optimism",
    "load_simulation_data",
    # Utils
    "EGAResult",
    "BootEGAResult",
    "embed",
    "to_igraph",
    "from_igraph",
]
