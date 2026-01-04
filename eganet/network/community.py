"""
Community detection algorithms.

Wraps NetworkX community detection algorithms and provides
EGAnet-specific handling for disconnected and singleton nodes.
"""

from __future__ import annotations
from typing import Union, Optional, Literal, Callable, Dict, Any, List
import numpy as np
import networkx as nx
from collections import Counter


CommunityAlgorithm = Literal[
    "walktrap", "louvain", "leiden", "fast_greedy", "label_prop",
    "leading_eigen", "infomap", "edge_betweenness"
]


def community_detection(
    network: np.ndarray,
    algorithm: Union[CommunityAlgorithm, Callable] = "walktrap",
    allow_singleton: bool = False,
    resolution: float = 1.0,
    seed: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Apply community detection algorithm.

    Parameters
    ----------
    network : np.ndarray
        Network adjacency matrix (can be weighted)
    algorithm : str or callable
        Community detection algorithm
    allow_singleton : bool
        Whether to allow singleton communities
    resolution : float
        Resolution parameter for modularity-based methods
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Community membership vector (1-indexed, NaN for disconnected)
    """
    network = np.abs(np.array(network))
    n_nodes = network.shape[0]

    node_strength = np.sum(network, axis=0) - np.diag(network)
    disconnected = node_strength == 0

    membership = np.full(n_nodes, np.nan)

    if np.all(disconnected):
        return membership

    G = _matrix_to_networkx(network)

    connected_nodes = np.where(~disconnected)[0]

    if len(connected_nodes) <= 1:
        membership[connected_nodes] = 1
        return membership

    subgraph = G.subgraph(connected_nodes)

    if isinstance(algorithm, str):
        algorithm = algorithm.lower()

        if algorithm == "walktrap":
            communities = _walktrap(subgraph, **kwargs)
        elif algorithm == "louvain":
            communities = _louvain(subgraph, resolution, seed, **kwargs)
        elif algorithm == "leiden":
            communities = _leiden(subgraph, resolution, seed, **kwargs)
        elif algorithm == "fast_greedy":
            communities = _fast_greedy(subgraph, **kwargs)
        elif algorithm == "label_prop":
            communities = _label_propagation(subgraph, seed, **kwargs)
        elif algorithm == "leading_eigen":
            communities = _leading_eigen(subgraph, **kwargs)
        elif algorithm == "infomap":
            communities = _louvain(subgraph, resolution, seed, **kwargs)
        elif algorithm == "edge_betweenness":
            communities = _edge_betweenness(subgraph, **kwargs)
        else:
            communities = _louvain(subgraph, resolution, seed, **kwargs)
    else:
        communities = algorithm(subgraph, **kwargs)

    for comm_id, nodes in enumerate(communities, start=1):
        for node in nodes:
            membership[node] = comm_id

    if not allow_singleton:
        counts = Counter(membership[~np.isnan(membership)])
        singletons = {c for c, count in counts.items() if count == 1}
        for singleton in singletons:
            membership[membership == singleton] = np.nan

    membership = _reindex_memberships(membership)

    return membership


def _matrix_to_networkx(network: np.ndarray) -> nx.Graph:
    """Convert adjacency matrix to NetworkX graph."""
    network = network.copy()
    np.fill_diagonal(network, 0)

    G = nx.Graph()
    n = network.shape[0]
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if network[i, j] != 0:
                G.add_edge(i, j, weight=abs(network[i, j]))

    return G


def _reindex_memberships(membership: np.ndarray) -> np.ndarray:
    """Reindex memberships to be sequential from 1."""
    result = membership.copy()
    valid = result[~np.isnan(result)]

    if len(valid) == 0:
        return result

    unique = np.unique(valid)
    mapping = {old: new + 1 for new, old in enumerate(sorted(unique))}

    for old, new in mapping.items():
        result[result == old] = new

    return result


def _walktrap(G: nx.Graph, steps: int = 4, **kwargs) -> List[set]:
    """Walktrap community detection using hierarchical clustering."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if len(G.nodes) < 2:
        return [set(G.nodes)]

    nodes = list(G.nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    adj = nx.to_numpy_array(G, nodelist=nodes)
    strength = adj.sum(axis=1)
    strength[strength == 0] = 1

    adj_normalized = adj.copy()
    np.fill_diagonal(adj_normalized, adj.max(axis=1))
    strength_with_diag = adj_normalized.sum(axis=1)
    P = adj_normalized / strength_with_diag[:, np.newaxis]

    P_steps = np.linalg.matrix_power(P, steps)

    D = np.sqrt(1.0 / strength)[:, np.newaxis]
    P_normalized = P_steps * D

    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = P_normalized[i] - P_normalized[j]
            distances[i, j] = distances[j, i] = np.sqrt(np.sum(diff ** 2))

    condensed = squareform(distances)
    Z = linkage(condensed, method='ward')

    best_modularity = -np.inf
    best_labels = np.ones(n, dtype=int)

    for k in range(1, min(n, 10) + 1):
        labels = fcluster(Z, k, criterion='maxclust')
        mod = _modularity(adj, labels)
        if mod > best_modularity:
            best_modularity = mod
            best_labels = labels

    communities = []
    for label in np.unique(best_labels):
        comm = {nodes[i] for i in range(n) if best_labels[i] == label}
        communities.append(comm)

    return communities


def _louvain(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: Optional[int] = None,
    **kwargs
) -> List[set]:
    """Louvain community detection."""
    try:
        from networkx.algorithms.community import louvain_communities
        return louvain_communities(G, resolution=resolution, seed=seed)
    except ImportError:
        return _greedy_modularity(G)


def _leiden(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: Optional[int] = None,
    **kwargs
) -> List[set]:
    """Leiden community detection (falls back to Louvain if not available)."""
    try:
        import leidenalg
        import igraph as ig

        edges = list(G.edges())
        weights = [G[u][v].get('weight', 1.0) for u, v in edges]

        ig_graph = ig.Graph(edges=edges, directed=False)
        ig_graph.es['weight'] = weights

        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
            seed=seed
        )

        communities = [set(comm) for comm in partition]
        return communities
    except ImportError:
        return _louvain(G, resolution, seed)


def _fast_greedy(G: nx.Graph, **kwargs) -> List[set]:
    """Fast greedy modularity optimization."""
    return _greedy_modularity(G)


def _greedy_modularity(G: nx.Graph) -> List[set]:
    """Greedy modularity optimization."""
    from networkx.algorithms.community import greedy_modularity_communities
    return list(greedy_modularity_communities(G))


def _label_propagation(
    G: nx.Graph,
    seed: Optional[int] = None,
    **kwargs
) -> List[set]:
    """Label propagation community detection."""
    from networkx.algorithms.community import label_propagation_communities
    return list(label_propagation_communities(G))


def _leading_eigen(G: nx.Graph, **kwargs) -> List[set]:
    """Leading eigenvector community detection."""
    from scipy.sparse.linalg import eigsh

    adj = nx.to_numpy_array(G)
    n = adj.shape[0]

    if n < 2:
        return [set(G.nodes)]

    degree = adj.sum(axis=1)
    m = degree.sum() / 2

    if m == 0:
        return [set(G.nodes)]

    B = adj - np.outer(degree, degree) / (2 * m)

    try:
        eigenvalues, eigenvectors = eigsh(B, k=1, which='LA')
        leading_vec = eigenvectors[:, 0]

        labels = (leading_vec > 0).astype(int)

        communities = []
        nodes = list(G.nodes)
        for label in [0, 1]:
            comm = {nodes[i] for i in range(n) if labels[i] == label}
            if comm:
                communities.append(comm)

        if len(communities) == 1:
            return communities

        return communities
    except Exception:
        return [set(G.nodes)]


def _edge_betweenness(G: nx.Graph, **kwargs) -> List[set]:
    """Edge betweenness community detection."""
    from networkx.algorithms.community import girvan_newman

    if len(G.edges) == 0:
        return [set(G.nodes)]

    communities_generator = girvan_newman(G)

    best_modularity = -np.inf
    best_partition = [set(G.nodes)]

    for i, partition in enumerate(communities_generator):
        if i >= 10:
            break

        partition_list = list(partition)

        adj = nx.to_numpy_array(G)
        labels = np.zeros(len(G.nodes))
        nodes = list(G.nodes)
        for comm_idx, comm in enumerate(partition_list):
            for node in comm:
                labels[nodes.index(node)] = comm_idx

        mod = _modularity(adj, labels)
        if mod > best_modularity:
            best_modularity = mod
            best_partition = partition_list

    return best_partition


def _modularity(adj: np.ndarray, labels: np.ndarray) -> float:
    """Compute modularity of a partition."""
    m = adj.sum() / 2
    if m == 0:
        return 0

    n = len(labels)
    degree = adj.sum(axis=1)

    Q = 0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Q += adj[i, j] - degree[i] * degree[j] / (2 * m)

    return Q / (2 * m)


def community_consensus(
    network: np.ndarray,
    algorithm: CommunityAlgorithm = "louvain",
    consensus_method: str = "most_common",
    consensus_iter: int = 1000,
    seed: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Consensus clustering using multiple runs.

    Parameters
    ----------
    network : np.ndarray
        Network adjacency matrix
    algorithm : str
        Community detection algorithm
    consensus_method : str
        Method for consensus ("most_common", "highest_modularity")
    consensus_iter : int
        Number of iterations
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Consensus community membership
    """
    n_nodes = network.shape[0]
    all_memberships = []

    rng = np.random.default_rng(seed)

    for i in range(consensus_iter):
        iter_seed = rng.integers(0, 2**31)
        membership = community_detection(
            network, algorithm, allow_singleton=True, seed=iter_seed, **kwargs
        )
        all_memberships.append(membership)

    memberships_array = np.array(all_memberships)

    if consensus_method == "most_common":
        final_membership = np.full(n_nodes, np.nan)

        for j in range(n_nodes):
            col = memberships_array[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                counts = Counter(valid)
                final_membership[j] = counts.most_common(1)[0][0]

    elif consensus_method == "highest_modularity":
        best_mod = -np.inf
        best_idx = 0

        G = _matrix_to_networkx(network)
        adj = nx.to_numpy_array(G)

        for i, mem in enumerate(all_memberships):
            labels = mem.copy()
            labels[np.isnan(labels)] = -1
            mod = _modularity(adj, labels)
            if mod > best_mod:
                best_mod = mod
                best_idx = i

        final_membership = all_memberships[best_idx]

    else:
        final_membership = all_memberships[0]

    final_membership = _reindex_memberships(final_membership)

    counts = Counter(final_membership[~np.isnan(final_membership)])
    singletons = {c for c, count in counts.items() if count == 1}
    for singleton in singletons:
        final_membership[final_membership == singleton] = np.nan

    return _reindex_memberships(final_membership)


def community_unidimensional(
    data: np.ndarray,
    network: np.ndarray,
    uni_method: str = "louvain",
    **kwargs
) -> np.ndarray:
    """
    Check for unidimensional structure.

    Parameters
    ----------
    data : np.ndarray
        Original data or correlation matrix
    network : np.ndarray
        Estimated network
    uni_method : str
        Method for unidimensionality check ("louvain", "le", "expand")

    Returns
    -------
    np.ndarray
        Community membership (all 1s if unidimensional)
    """
    n_nodes = network.shape[0]

    if uni_method.lower() == "louvain":
        membership = community_detection(network, "louvain", **kwargs)
        n_communities = len(np.unique(membership[~np.isnan(membership)]))

        if n_communities == 1:
            return np.ones(n_nodes)
        else:
            return community_detection(network, "walktrap", **kwargs)

    elif uni_method.lower() == "le":
        membership = community_detection(network, "leading_eigen", **kwargs)
        n_communities = len(np.unique(membership[~np.isnan(membership)]))

        if n_communities == 1:
            return np.ones(n_nodes)
        else:
            return community_detection(network, "walktrap", **kwargs)

    elif uni_method.lower() == "expand":
        from eganet.utils.helpers import is_correlation_matrix

        if is_correlation_matrix(data):
            corr = data
        else:
            from eganet.correlation.auto import auto_correlate
            corr, _ = auto_correlate(data, **kwargs)

        n_expand = 4
        expanded = np.zeros((n_nodes + n_expand, n_nodes + n_expand))
        expanded[:n_nodes, :n_nodes] = corr

        for i in range(n_expand):
            for j in range(n_expand):
                if i != j:
                    expanded[n_nodes + i, n_nodes + j] = 0.5
            expanded[n_nodes + i, n_nodes + i] = 1.0

        from eganet.network.estimation import glasso
        expanded_network = glasso(expanded, n=100)

        membership = community_detection(expanded_network, "walktrap")
        n_communities = len(np.unique(membership[~np.isnan(membership)]))

        if n_communities <= 2:
            return np.ones(n_nodes)
        else:
            return community_detection(network, "walktrap", **kwargs)

    return community_detection(network, "walktrap", **kwargs)
