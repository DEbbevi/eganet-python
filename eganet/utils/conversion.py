"""
Conversion utilities between network formats.
"""

from __future__ import annotations
from typing import Union, Optional
import numpy as np


def convert_to_networkx(network: np.ndarray):
    """
    Convert adjacency matrix to NetworkX graph.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix

    Returns
    -------
    networkx.Graph
        NetworkX graph object
    """
    import networkx as nx

    network = np.array(network)
    np.fill_diagonal(network, 0)

    G = nx.Graph()
    n = network.shape[0]
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if network[i, j] != 0:
                G.add_edge(i, j, weight=network[i, j])

    return G


def networkx_to_matrix(G) -> np.ndarray:
    """
    Convert NetworkX graph to adjacency matrix.

    Parameters
    ----------
    G : networkx.Graph
        NetworkX graph

    Returns
    -------
    np.ndarray
        Adjacency matrix
    """
    import networkx as nx
    return nx.to_numpy_array(G)


def convert_to_igraph(network: np.ndarray):
    """
    Convert adjacency matrix to igraph graph.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix

    Returns
    -------
    igraph.Graph
        igraph graph object
    """
    try:
        import igraph as ig
    except ImportError:
        raise ImportError("igraph is required for this function")

    network = np.array(network)
    np.fill_diagonal(network, 0)

    n = network.shape[0]
    edges = []
    weights = []

    for i in range(n):
        for j in range(i + 1, n):
            if network[i, j] != 0:
                edges.append((i, j))
                weights.append(abs(network[i, j]))

    G = ig.Graph(n=n, edges=edges, directed=False)
    G.es['weight'] = weights

    return G


def igraph_to_matrix(G) -> np.ndarray:
    """
    Convert igraph graph to adjacency matrix.

    Parameters
    ----------
    G : igraph.Graph
        igraph graph

    Returns
    -------
    np.ndarray
        Adjacency matrix
    """
    return np.array(G.get_adjacency(attribute='weight').data)


def sparse_to_dense(sparse_matrix) -> np.ndarray:
    """
    Convert sparse matrix to dense numpy array.

    Parameters
    ----------
    sparse_matrix : scipy.sparse matrix
        Sparse matrix

    Returns
    -------
    np.ndarray
        Dense array
    """
    from scipy import sparse

    if sparse.issparse(sparse_matrix):
        return sparse_matrix.toarray()
    return np.array(sparse_matrix)


def dense_to_sparse(matrix: np.ndarray, threshold: float = 0):
    """
    Convert dense matrix to sparse format.

    Parameters
    ----------
    matrix : np.ndarray
        Dense matrix
    threshold : float
        Values below threshold become zero

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix
    """
    from scipy import sparse

    matrix = matrix.copy()
    matrix[np.abs(matrix) <= threshold] = 0

    return sparse.csr_matrix(matrix)


to_igraph = convert_to_igraph
from_igraph = igraph_to_matrix
