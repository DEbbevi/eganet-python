"""Correlation computation methods."""

from eganet.correlation.auto import auto_correlate
from eganet.correlation.polychoric import polychoric_matrix
from eganet.correlation.cosine import cosine_similarity

__all__ = ["auto_correlate", "polychoric_matrix", "cosine_similarity"]
