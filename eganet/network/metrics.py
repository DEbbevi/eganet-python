"""
Network metrics and properties.
"""

from __future__ import annotations
from typing import Union, Optional, Dict
import numpy as np


def modularity(network: np.ndarray, memberships: np.ndarray) -> float:
    """
    Compute modularity of network partition.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix
    memberships : np.ndarray
        Community memberships

    Returns
    -------
    float
        Modularity value
    """
    abs_network = np.abs(network)
    np.fill_diagonal(abs_network, 0)

    m = np.sum(abs_network) / 2
    if m == 0:
        return 0.0

    n = network.shape[0]
    strength = np.sum(abs_network, axis=1)

    Q = 0.0
    for i in range(n):
        for j in range(n):
            if not np.isnan(memberships[i]) and not np.isnan(memberships[j]):
                if memberships[i] == memberships[j]:
                    Q += abs_network[i, j] - strength[i] * strength[j] / (2 * m)

    return Q / (2 * m)


def weighted_topological_overlap(network: np.ndarray) -> np.ndarray:
    """
    Compute weighted topological overlap matrix.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix

    Returns
    -------
    np.ndarray
        WTO matrix
    """
    from eganet.psychometrics.uva import weighted_topological_overlap as wto
    return wto(network)


def frobenius_norm(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    Compute Frobenius norm of difference between matrices.

    Parameters
    ----------
    matrix1 : np.ndarray
        First matrix
    matrix2 : np.ndarray
        Second matrix

    Returns
    -------
    float
        Frobenius norm
    """
    diff = matrix1 - matrix2
    return np.sqrt(np.sum(diff ** 2))


def node_strength(network: np.ndarray) -> np.ndarray:
    """
    Compute node strength (weighted degree).

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix

    Returns
    -------
    np.ndarray
        Node strengths
    """
    abs_network = np.abs(network)
    np.fill_diagonal(abs_network, 0)
    return np.sum(abs_network, axis=1)


def clustering_coefficient(network: np.ndarray) -> np.ndarray:
    """
    Compute local clustering coefficient.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix

    Returns
    -------
    np.ndarray
        Clustering coefficients per node
    """
    abs_network = np.abs(network)
    np.fill_diagonal(abs_network, 0)
    n = network.shape[0]

    cc = np.zeros(n)

    for i in range(n):
        neighbors = np.where(abs_network[i] > 0)[0]
        k = len(neighbors)

        if k < 2:
            cc[i] = 0
        else:
            subgraph = abs_network[np.ix_(neighbors, neighbors)]
            n_edges = np.sum(subgraph > 0) / 2
            cc[i] = 2 * n_edges / (k * (k - 1))

    return cc


def betweenness_centrality(network: np.ndarray) -> np.ndarray:
    """
    Compute betweenness centrality.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix

    Returns
    -------
    np.ndarray
        Betweenness centrality per node
    """
    from eganet.utils.conversion import convert_to_networkx
    import networkx as nx

    G = convert_to_networkx(network)
    bc = nx.betweenness_centrality(G, weight="weight")

    n = network.shape[0]
    result = np.zeros(n)
    for node, value in bc.items():
        result[node] = value

    return result


def closeness_centrality(network: np.ndarray) -> np.ndarray:
    """
    Compute closeness centrality.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix

    Returns
    -------
    np.ndarray
        Closeness centrality per node
    """
    from eganet.utils.conversion import convert_to_networkx
    import networkx as nx

    G = convert_to_networkx(network)

    inv_weight = {}
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        inv_weight[(u, v)] = 1 / w if w > 0 else float("inf")

    nx.set_edge_attributes(G, inv_weight, "distance")

    cc = nx.closeness_centrality(G, distance="distance")

    n = network.shape[0]
    result = np.zeros(n)
    for node, value in cc.items():
        result[node] = value

    return result


def network_density(network: np.ndarray) -> float:
    """
    Compute network density.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix

    Returns
    -------
    float
        Network density
    """
    abs_network = np.abs(network)
    np.fill_diagonal(abs_network, 0)

    n = network.shape[0]
    n_possible = n * (n - 1) / 2
    n_edges = np.sum(abs_network > 0) / 2

    return n_edges / n_possible if n_possible > 0 else 0


def average_path_length(network: np.ndarray) -> float:
    """
    Compute average shortest path length.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix

    Returns
    -------
    float
        Average path length
    """
    from eganet.utils.conversion import convert_to_networkx
    import networkx as nx

    G = convert_to_networkx(network)

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        G = G.subgraph(largest)

    try:
        return nx.average_shortest_path_length(G)
    except Exception:
        return float("inf")


def small_worldness(network: np.ndarray, n_random: int = 100) -> float:
    """
    Compute small-worldness index.

    Parameters
    ----------
    network : np.ndarray
        Adjacency matrix
    n_random : int
        Number of random networks for comparison

    Returns
    -------
    float
        Small-worldness index (sigma)
    """
    C = np.mean(clustering_coefficient(network))
    L = average_path_length(network)

    n = network.shape[0]
    density = network_density(network)

    C_random = []
    L_random = []

    for _ in range(n_random):
        from eganet.simulation.known_graph import random_graph
        random_net = random_graph(n, density)
        C_random.append(np.mean(clustering_coefficient(random_net)))
        L_random.append(average_path_length(random_net))

    C_rand = np.mean(C_random)
    L_rand = np.mean(L_random)

    if C_rand == 0 or L == 0:
        return 0

    gamma = C / C_rand
    lambda_val = L / L_rand

    if lambda_val == 0:
        return 0

    return gamma / lambda_val
