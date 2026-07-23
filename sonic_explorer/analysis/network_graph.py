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

build_metadata_similarity_graph() is the naive baseline for Overview section
1.1 -- deliberately the *strongest* defensible non-audio baseline, not a
genre-only strawman: it averages four independently-computed [0, 1]
similarity signals (genre_top match, FMA's fuller genres_all overlap, same-
album membership, free-text tag overlap), none of which involve listening to
the audio at all. It reuses the exact same k-NN + spring-layout + clustering
pipeline as the real graph rather than connecting every same-genre pair as a
full clique: a literal clique scales O(songs-per-genre^2) and would produce
~120k edges at this project's full ~1400-song library (unrenderable), while
k-NN sampling stays O(songs * k) at any library size and still shows the
thing the comparison is meant to show -- mostly genre-clustered songs with
occasional real cross-genre edges (shared tag, shared album) the audio graph
is free to draw far more of.

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


@dataclass
class SongMetadata:
    genre_top: str
    genres_all: frozenset = frozenset()  # FMA sub-genre IDs (int)
    album_id: int | None = None
    tags: frozenset = frozenset()  # uploader-supplied free text (str)


def _jaccard_similarity_matrix(sets: list[frozenset], vocabulary: list) -> np.ndarray:
    """Vectorized Jaccard overlap over a shared multi-hot encoding -- an n^2
    matrix multiply, not a Python double loop, so this stays cheap at library
    scale. Two empty sets never count as similar (0/0 -> 0, not 1): "neither
    song has any tags" isn't evidence they're alike."""
    vocab_index = {item: i for i, item in enumerate(vocabulary)}
    multi_hot = np.zeros((len(sets), len(vocabulary)))
    for i, s in enumerate(sets):
        for item in s:
            multi_hot[i, vocab_index[item]] = 1.0
    intersection = multi_hot @ multi_hot.T
    row_sums = multi_hot.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _exact_match_similarity_matrix(values: list) -> np.ndarray:
    """1.0 where two songs share the same non-null value (e.g. album_id),
    0.0 otherwise -- two songs both missing the value must never count as a
    match, so each None is remapped to its own unique sentinel first."""
    resolved = []
    next_sentinel = -1
    for v in values:
        if v is None:
            resolved.append(next_sentinel)
            next_sentinel -= 1
        else:
            resolved.append(v)
    arr = np.array(resolved).reshape(-1, 1)
    return (arr == arr.T).astype(float)


def build_metadata_similarity_graph(
    song_metadata: dict[int, SongMetadata],
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    random_state: int = 42,
) -> NetworkGraphResult:
    """No single metadata field is a fair audio-free baseline on its own
    (genre_top alone is a strawman), so this equal-weight-averages four
    independently-computed [0, 1] similarity signals -- genre_top match,
    genres_all overlap, same-album membership, tag overlap -- the same
    blend-independent-[0,1]-scores approach build_blended_similarity_graph
    already uses for audio facets (see module docstring), reused here for a
    second kind of heterogeneous signal rather than inventing a new design.
    Equal weights, not hand-tuned ones: nothing here was picked to nudge the
    outcome toward a particular story. A signal that's empty across the
    whole library (e.g. no song has recovered tag data) safely contributes a
    zero matrix rather than distorting the average.

    Clusters are still genre_top (via the same one-hot-KMeans trick as the
    single-signal version this replaced), purely so every graph on the page
    uses consistent, readable color-coding -- edges are free to cross those
    colors now (a shared tag or album can connect two genres), which is the
    point: any cross-genre edge here is real recovered metadata signal, not
    a structural impossibility the way it was for a genre-only baseline."""
    song_ids = list(song_metadata.keys())
    if not song_ids:
        return NetworkGraphResult(nodes=[], edges=[])

    genres = sorted({m.genre_top for m in song_metadata.values()})
    genre_index = {g: i for i, g in enumerate(genres)}
    genre_one_hot = np.zeros((len(song_ids), len(genres)))
    for i, sid in enumerate(song_ids):
        genre_one_hot[i, genre_index[song_metadata[sid].genre_top]] = 1.0

    if len(song_ids) < 2:
        return _graph_from_similarity(song_ids, None, genre_one_hot, k_neighbors, len(genres), random_state)

    genre_sim = genre_one_hot @ genre_one_hot.T

    all_sub_genres = sorted({g for m in song_metadata.values() for g in m.genres_all})
    genres_all_sim = (
        _jaccard_similarity_matrix([song_metadata[sid].genres_all for sid in song_ids], all_sub_genres)
        if all_sub_genres
        else np.zeros((len(song_ids), len(song_ids)))
    )

    album_sim = _exact_match_similarity_matrix([song_metadata[sid].album_id for sid in song_ids])

    all_tags = sorted({t for m in song_metadata.values() for t in m.tags})
    tags_sim = (
        _jaccard_similarity_matrix([song_metadata[sid].tags for sid in song_ids], all_tags)
        if all_tags
        else np.zeros((len(song_ids), len(song_ids)))
    )

    combined_sims = (genre_sim + genres_all_sim + album_sim + tags_sim) / 4.0
    return _graph_from_similarity(song_ids, combined_sims, genre_one_hot, k_neighbors, len(genres), random_state)


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
