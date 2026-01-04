"""
Generate known graph structures.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def known_graph(
    structure: str = "random",
    n: int = 10,
    p: float = 0.3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate known graph structure.

    Parameters
    ----------
    structure : str
        Graph structure type
    n : int
        Number of nodes
    p : float
        Edge probability or density
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Adjacency matrix
    """
    rng = np.random.default_rng(seed)

    if structure == "random":
        return random_graph(n, p, rng)
    elif structure == "scale_free":
        return scale_free_graph(n, rng)
    elif structure == "small_world":
        return small_world_graph(n, k=4, p=p, rng=rng)
    elif structure == "ring":
        return ring_graph(n)
    elif structure == "star":
        return star_graph(n)
    elif structure == "complete":
        return complete_graph(n)
    elif structure == "band":
        return band_graph(n, bandwidth=2)
    else:
        return random_graph(n, p, rng)


def random_graph(n: int, p: float, rng=None) -> np.ndarray:
    """Erdos-Renyi random graph."""
    if rng is None:
        rng = np.random.default_rng()

    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                weight = rng.uniform(0.3, 0.7)
                adj[i, j] = weight
                adj[j, i] = weight
    return adj


def scale_free_graph(n: int, rng=None) -> np.ndarray:
    """Barabasi-Albert scale-free graph."""
    if rng is None:
        rng = np.random.default_rng()

    adj = np.zeros((n, n))
    m = 2

    for i in range(m):
        for j in range(i + 1, m):
            weight = rng.uniform(0.3, 0.7)
            adj[i, j] = weight
            adj[j, i] = weight

    for i in range(m, n):
        degrees = np.sum(adj > 0, axis=1)[:i]
        if np.sum(degrees) == 0:
            probs = np.ones(i) / i
        else:
            probs = degrees / np.sum(degrees)

        targets = rng.choice(i, size=min(m, i), replace=False, p=probs)
        for t in targets:
            weight = rng.uniform(0.3, 0.7)
            adj[i, t] = weight
            adj[t, i] = weight

    return adj


def small_world_graph(n: int, k: int = 4, p: float = 0.1, rng=None) -> np.ndarray:
    """Watts-Strogatz small-world graph."""
    if rng is None:
        rng = np.random.default_rng()

    adj = np.zeros((n, n))

    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % n
            weight = rng.uniform(0.3, 0.7)
            adj[i, neighbor] = weight
            adj[neighbor, i] = weight

    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < p:
                neighbor = (i + j) % n
                if adj[i, neighbor] > 0:
                    adj[i, neighbor] = 0
                    adj[neighbor, i] = 0

                    new_neighbor = rng.integers(n)
                    while new_neighbor == i or adj[i, new_neighbor] > 0:
                        new_neighbor = rng.integers(n)

                    weight = rng.uniform(0.3, 0.7)
                    adj[i, new_neighbor] = weight
                    adj[new_neighbor, i] = weight

    return adj


def ring_graph(n: int) -> np.ndarray:
    """Ring/cycle graph."""
    adj = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = 0.5
        adj[j, i] = 0.5
    return adj


def star_graph(n: int) -> np.ndarray:
    """Star graph with center at node 0."""
    adj = np.zeros((n, n))
    for i in range(1, n):
        adj[0, i] = 0.5
        adj[i, 0] = 0.5
    return adj


def complete_graph(n: int) -> np.ndarray:
    """Complete graph."""
    adj = np.ones((n, n)) * 0.5
    np.fill_diagonal(adj, 0)
    return adj


def band_graph(n: int, bandwidth: int = 2) -> np.ndarray:
    """Band/diagonal graph."""
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            if i != j:
                adj[i, j] = 0.5
    return adj


def community_graph(
    community_sizes: list,
    p_in: float = 0.7,
    p_out: float = 0.1,
    seed: Optional[int] = None
) -> tuple:
    """
    Generate graph with community structure.

    Parameters
    ----------
    community_sizes : list
        Size of each community
    p_in : float
        Within-community edge probability
    p_out : float
        Between-community edge probability
    seed : int, optional
        Random seed

    Returns
    -------
    tuple
        (adjacency matrix, community labels)
    """
    rng = np.random.default_rng(seed)
    n = sum(community_sizes)

    labels = np.zeros(n, dtype=int)
    idx = 0
    for comm, size in enumerate(community_sizes):
        labels[idx:idx + size] = comm
        idx += size

    adj = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            same_community = labels[i] == labels[j]
            prob = p_in if same_community else p_out

            if rng.random() < prob:
                weight = rng.uniform(0.3, 0.7)
                adj[i, j] = weight
                adj[j, i] = weight

    return adj, labels
