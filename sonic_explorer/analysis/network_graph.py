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

build_genre_similarity_graph() is the naive baseline for Overview section
1.1: "similarity" here is a genre indicator (1.0 same genre, 0.0 otherwise)
instead of cosine similarity over audio embeddings -- no audio analysis at
all. It reuses the exact same k-NN + spring-layout + clustering pipeline as
the real graph rather than connecting every same-genre pair as a full clique:
a literal clique scales O(songs-per-genre^2) and would produce ~120k edges
at this project's full ~1400-song library (unrenderable), while k-NN sampling
stays O(songs * k) at any library size and still shows the thing the
comparison is meant to show -- fully genre-siloed islands with zero
cross-genre edges, next to the real graph's cross-genre bridges.

build_blended_similarity_graph() supports picking several facets at once
(e.g. sound + vocal): it blends by averaging each facet's independently-
computed cosine-similarity matrix, not by averaging raw vectors -- different
facets live in different, often differently-sized embedding spaces (sound is
512-dim CLAP, harmony is 24-dim chroma, ...), so there's no shared vector
space to average within. Averaging similarity scores instead sidesteps that
entirely: every facet always produces a similarity in [0, 1] regardless of
its native dimensionality, so those are safe to combine. Both functions
share the same edge/layout/clustering construction (_graph_from_similarity).

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


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return normalized @ normalized.T


def _graph_from_similarity(
    song_ids: list[int],
    sims: np.ndarray | None,
    cluster_matrix: np.ndarray,
    k_neighbors: int,
    n_clusters: int,
    random_state: int,
) -> NetworkGraphResult:
    """Shared edge-construction (k-NN over a precomputed similarity matrix),
    force-directed layout, and clustering -- used by both a single facet's
    own cosine-similarity matrix and a multi-facet blended one, so "pick one
    facet" and "pick several and blend" are the same graph-building logic
    with a different similarity matrix in, not two implementations."""
    graph = nx.Graph()
    graph.add_nodes_from(song_ids)

    if len(song_ids) >= 2 and sims is not None:
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
        labels = KMeans(
            n_clusters=max(1, min(n_clusters, len(song_ids))), random_state=random_state, n_init=10
        ).fit_predict(cluster_matrix)
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
    sims = _cosine_similarity_matrix(matrix) if len(song_ids) >= 2 else None
    return _graph_from_similarity(song_ids, sims, matrix, k_neighbors, n_clusters, random_state)


def build_genre_similarity_graph(
    song_genres: dict[int, str],
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    random_state: int = 42,
) -> NetworkGraphResult:
    """The naive, audio-free baseline: songs are "similar" iff they share a
    genre tag. song_genres is {song_id: genre_top}. Clusters are the genres
    themselves (one-hot vectors of the same genre are identical points, so
    KMeans with n_clusters = number of distinct genres always separates them
    perfectly -- no coincidental overlap to worry about)."""
    song_ids = list(song_genres.keys())
    if not song_ids:
        return NetworkGraphResult(nodes=[], edges=[])

    genres = sorted({g for g in song_genres.values()})
    genre_index = {g: i for i, g in enumerate(genres)}
    one_hot = np.zeros((len(song_ids), len(genres)))
    for i, sid in enumerate(song_ids):
        one_hot[i, genre_index[song_genres[sid]]] = 1.0

    sims = one_hot @ one_hot.T if len(song_ids) >= 2 else None
    return _graph_from_similarity(song_ids, sims, one_hot, k_neighbors, len(genres), random_state)


def build_blended_similarity_graph(
    song_vectors_by_facet: dict[str, dict[int, np.ndarray]],
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
) -> NetworkGraphResult:
    """song_vectors_by_facet: {"sound": {song_id: vec, ...}, "vocal": {...}, ...}.
    Only songs present in every selected facet participate -- a song missing
    one of the chosen facets has no similarity score to contribute for it, so
    it can't be fairly blended in (better than silently treating a missing
    facet as "0 similarity to everyone," which would bias the blend)."""
    if not song_vectors_by_facet:
        return NetworkGraphResult(nodes=[], edges=[])

    common_ids: set[int] | None = None
    for vectors in song_vectors_by_facet.values():
        ids = set(vectors.keys())
        common_ids = ids if common_ids is None else common_ids & ids
    song_ids = sorted(common_ids) if common_ids else []
    if not song_ids:
        return NetworkGraphResult(nodes=[], edges=[])

    if len(song_ids) < 2:
        cluster_matrix = np.zeros((len(song_ids), 1))
        return _graph_from_similarity(song_ids, None, cluster_matrix, k_neighbors, n_clusters, random_state)

    blended_sims = None
    normalized_blocks = []
    for vectors in song_vectors_by_facet.values():
        matrix = np.stack([vectors[sid] for sid in song_ids])
        sims = _cosine_similarity_matrix(matrix)
        blended_sims = sims if blended_sims is None else blended_sims + sims
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_blocks.append(matrix / norms)  # unit-normalized so no facet dominates clustering by raw scale
    blended_sims /= len(song_vectors_by_facet)
    cluster_matrix = np.concatenate(normalized_blocks, axis=1)

    return _graph_from_similarity(song_ids, blended_sims, cluster_matrix, k_neighbors, n_clusters, random_state)
