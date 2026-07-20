import numpy as np
import pytest

from sonic_explorer.analysis.network_graph import build_blended_similarity_graph, build_similarity_graph


def test_build_similarity_graph_handles_empty_input():
    result = build_similarity_graph({})
    assert result.nodes == []
    assert result.edges == []


def test_build_similarity_graph_handles_single_song():
    result = build_similarity_graph({1: np.array([1.0, 2.0, 3.0])})
    assert len(result.nodes) == 1
    assert result.nodes[0].song_id == 1
    assert result.edges == []


def test_build_similarity_graph_separates_distinct_clusters():
    rng = np.random.default_rng(0)
    cluster_a = {i: rng.normal(loc=[10, 10], scale=0.1) for i in range(6)}
    cluster_b = {i + 100: rng.normal(loc=[-10, -10], scale=0.1) for i in range(6)}
    song_vectors = {**cluster_a, **cluster_b}

    result = build_similarity_graph(song_vectors, k_neighbors=2, n_clusters=2)

    assert len(result.nodes) == 12
    labels_a = {n.cluster for n in result.nodes if n.song_id < 100}
    labels_b = {n.cluster for n in result.nodes if n.song_id >= 100}
    assert len(labels_a) == 1
    assert len(labels_b) == 1
    assert labels_a != labels_b


def test_build_similarity_graph_edges_stay_within_clusters_when_well_separated():
    """A k-NN graph over two far-apart, tight clusters should never connect
    across clusters -- every node's nearest neighbors are all in its own
    cluster given how far apart they are."""
    rng = np.random.default_rng(1)
    cluster_a = {i: rng.normal(loc=[50, 50], scale=0.1) for i in range(6)}
    cluster_b = {i + 100: rng.normal(loc=[-50, -50], scale=0.1) for i in range(6)}
    song_vectors = {**cluster_a, **cluster_b}

    result = build_similarity_graph(song_vectors, k_neighbors=2, n_clusters=2)

    for edge in result.edges:
        assert (edge.song_id_a < 100) == (edge.song_id_b < 100)


def test_build_similarity_graph_no_self_loops_or_duplicate_edges():
    rng = np.random.default_rng(2)
    song_vectors = {i: rng.normal(size=8) for i in range(20)}

    result = build_similarity_graph(song_vectors, k_neighbors=3)

    seen = set()
    for edge in result.edges:
        assert edge.song_id_a != edge.song_id_b
        key = frozenset((edge.song_id_a, edge.song_id_b))
        assert key not in seen
        seen.add(key)


def test_build_similarity_graph_edge_weights_in_range():
    rng = np.random.default_rng(3)
    song_vectors = {i: rng.normal(size=6) for i in range(15)}

    result = build_similarity_graph(song_vectors, k_neighbors=3)

    for edge in result.edges:
        assert 0.0 <= edge.weight <= 1.0 + 1e-9


def test_build_similarity_graph_respects_k_neighbors_upper_bound():
    """k_neighbors larger than the available pool must not crash -- clamps to
    n-1 (every other song becomes a neighbor)."""
    song_vectors = {i: np.array([float(i), 0.0]) for i in range(3)}

    result = build_similarity_graph(song_vectors, k_neighbors=50)

    assert len(result.nodes) == 3
    # with only 3 songs and k clamped to 2, the graph should be fully connected
    assert len(result.edges) == 3


def test_build_similarity_graph_all_nodes_present_even_if_isolated_edges_dont_duplicate():
    rng = np.random.default_rng(4)
    song_vectors = {i: rng.normal(size=4) for i in range(10)}

    result = build_similarity_graph(song_vectors, k_neighbors=2)

    node_ids = {n.song_id for n in result.nodes}
    assert node_ids == set(song_vectors.keys())
    for edge in result.edges:
        assert edge.song_id_a in node_ids
        assert edge.song_id_b in node_ids


def test_build_blended_similarity_graph_handles_empty_input():
    result = build_blended_similarity_graph({})
    assert result.nodes == []
    assert result.edges == []


def test_build_blended_similarity_graph_handles_no_common_songs():
    result = build_blended_similarity_graph({
        "sound": {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])},
        "harmony": {3: np.array([1.0, 0.0]), 4: np.array([0.0, 1.0])},
    })
    assert result.nodes == []
    assert result.edges == []


def test_build_blended_similarity_graph_only_includes_songs_present_in_every_facet():
    result = build_blended_similarity_graph({
        "sound": {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0]), 3: np.array([1.0, 1.0])},
        "harmony": {1: np.array([0.5, 0.5]), 2: np.array([0.2, 0.8])},  # song 3 missing here
    })
    node_ids = {n.song_id for n in result.nodes}
    assert node_ids == {1, 2}


def test_build_blended_similarity_graph_single_facet_matches_build_similarity_graph():
    """Blending with exactly one facet should be equivalent to not blending
    at all -- a sanity check that averaging a single similarity matrix is a
    no-op, not an accidental transformation."""
    rng = np.random.default_rng(5)
    vectors = {i: rng.normal(size=6) for i in range(8)}

    single = build_similarity_graph(vectors, k_neighbors=3, random_state=1)
    blended = build_blended_similarity_graph({"sound": vectors}, k_neighbors=3, random_state=1)

    single_edges = {frozenset((e.song_id_a, e.song_id_b)) for e in single.edges}
    blended_edges = {frozenset((e.song_id_a, e.song_id_b)) for e in blended.edges}
    assert single_edges == blended_edges


def test_build_blended_similarity_graph_separates_clusters_using_combined_signal():
    rng = np.random.default_rng(6)
    cluster_a_sound = {i: rng.normal(loc=[10, 10], scale=0.1) for i in range(6)}
    cluster_b_sound = {i + 100: rng.normal(loc=[-10, -10], scale=0.1) for i in range(6)}
    cluster_a_harmony = {i: rng.normal(loc=[5, 5], scale=0.1) for i in range(6)}
    cluster_b_harmony = {i + 100: rng.normal(loc=[-5, -5], scale=0.1) for i in range(6)}

    result = build_blended_similarity_graph(
        {
            "sound": {**cluster_a_sound, **cluster_b_sound},
            "harmony": {**cluster_a_harmony, **cluster_b_harmony},
        },
        k_neighbors=2, n_clusters=2,
    )

    assert len(result.nodes) == 12
    labels_a = {n.cluster for n in result.nodes if n.song_id < 100}
    labels_b = {n.cluster for n in result.nodes if n.song_id >= 100}
    assert len(labels_a) == 1
    assert len(labels_b) == 1
    assert labels_a != labels_b


def test_build_blended_similarity_graph_handles_single_common_song():
    result = build_blended_similarity_graph({
        "sound": {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])},
        "harmony": {1: np.array([0.5, 0.5])},  # only song 1 in common
    })
    assert len(result.nodes) == 1
    assert result.nodes[0].song_id == 1
    assert result.edges == []
