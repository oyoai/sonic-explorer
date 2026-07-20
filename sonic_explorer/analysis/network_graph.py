"""Song-as-node similarity graph over per-song mean-pooled facet embeddings --
the "relationship/network graph" alternative exploration mode (spec 2.1),
alongside the PCA/ICA spatial Taste Map. Same underlying song vectors as
taste_map.py (mean_pool_song_vectors), different visual metaphor: a
k-nearest-neighbor graph laid out with a force-directed algorithm
(networkx.spring_layout) instead of a PCA/ICA projection -- "start at one
song, see its direct neighbors, follow an edge to a neighbor's neighbors"
only makes sense with actual graph edges, which a scatter plot doesn't have.

Explore (global) and My Library both call build_similarity_graph() with a
different song_vectors dict (all songs vs. only saved ones) -- one
implementation, filtered by what's passed in, not a second code path.

Plain Python, no Streamlit import -- see spec 8.3's core/interface
separation, same discipline as taste_map.py."""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

DEFAULT_K_NEIGHBORS = 4
DEFAULT_N_CLUSTERS = 8


@dataclass
class GraphNode:
    song_id: int
    x: float
    y: float
    cluster: int


@dataclass
class GraphEdge:
    song_id_a: int
    song_id_b: int
    weight: float  # cosine similarity, clamped to [0, 1]


@dataclass
class NetworkGraphResult:
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def build_similarity_graph(
    song_vectors: dict[int, np.ndarray],
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
) -> NetworkGraphResult:
    song_ids = list(song_vectors.keys())
    if not song_ids:
        return NetworkGraphResult(nodes=[], edges=[])

    matrix = np.stack([song_vectors[sid] for sid in song_ids])

    graph = nx.Graph()
    graph.add_nodes_from(song_ids)

    if len(song_ids) >= 2:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        sims = normalized @ normalized.T

        k = max(1, min(k_neighbors, len(song_ids) - 1))
        for i, sid in enumerate(song_ids):
            row = sims[i].copy()
            row[i] = -np.inf  # exclude self from its own neighbor search
            neighbor_idx = np.argpartition(row, -k)[-k:]
            for j in neighbor_idx:
                other_sid = song_ids[j]
                weight = float(max(0.0, sims[i, j]))
                if not graph.has_edge(sid, other_sid):
                    graph.add_edge(sid, other_sid, weight=weight)

        positions = nx.spring_layout(graph, seed=random_state, weight="weight")
        labels = KMeans(n_clusters=max(1, min(n_clusters, len(song_ids))), random_state=random_state, n_init=10).fit_predict(matrix)
    else:
        positions = {sid: (0.0, 0.0) for sid in song_ids}
        labels = np.zeros(len(song_ids), dtype=int)

    nodes = [
        GraphNode(song_id=sid, x=float(positions[sid][0]), y=float(positions[sid][1]), cluster=int(labels[i]))
        for i, sid in enumerate(song_ids)
    ]
    edges = [
        GraphEdge(song_id_a=a, song_id_b=b, weight=float(data["weight"]))
        for a, b, data in graph.edges(data=True)
    ]
    return NetworkGraphResult(nodes=nodes, edges=edges)
