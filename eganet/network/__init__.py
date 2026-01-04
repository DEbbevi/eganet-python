"""Network estimation and community detection."""

from eganet.network.estimation import network_estimation, glasso, tmfg
from eganet.network.community import community_detection, community_consensus
from eganet.network.metrics import (
    modularity,
    weighted_topological_overlap as wto,
    frobenius_norm,
    node_strength,
    clustering_coefficient,
)
from eganet.network.comparison import network_compare
from eganet.network.regularization import network_regularization
from eganet.network.fit import network_fit, partial_to_correlation
from eganet.network.predictability import network_predictability

modularity_q = modularity

__all__ = [
    "network_estimation",
    "glasso",
    "tmfg",
    "community_detection",
    "community_consensus",
    "modularity",
    "modularity_q",
    "wto",
    "frobenius_norm",
    "node_strength",
    "clustering_coefficient",
    "network_compare",
    "network_regularization",
    "network_fit",
    "partial_to_correlation",
    "network_predictability",
]
