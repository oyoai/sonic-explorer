import numpy as np
import pytest

from sonic_explorer.analysis.network_graph import (
    DEFAULT_METADATA_WEIGHTS,
    GraphEdge,
    SongMetadata,
    build_blended_similarity_graph,
    build_metadata_similarity_graph,
    build_similarity_graph,
    combine_metadata_similarities,
    compute_metadata_similarity_components,
    cosine_similarity_between,
    cross_genre_edge_fraction,
    pick_metadata_mismatch_pair,
    pick_real_cross_genre_pair,
)


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


def test_build_metadata_similarity_graph_handles_empty_input():
    result = build_metadata_similarity_graph({})
    assert result.nodes == []
    assert result.edges == []


def test_build_metadata_similarity_graph_handles_single_song():
    result = build_metadata_similarity_graph({1: SongMetadata(genre_top="Rock")})
    assert len(result.nodes) == 1
    assert result.nodes[0].song_id == 1
    assert result.edges == []


def test_build_metadata_similarity_graph_genre_only_still_isolates_by_genre():
    """With no genres_all/album/tags data at all (the common case until
    enrichment has run), this must behave exactly like the old genre-only
    baseline: zero cross-genre edges."""
    metadata = {i: SongMetadata(genre_top="Rock") for i in range(10)} | {
        i + 100: SongMetadata(genre_top="Jazz") for i in range(10)
    }

    result = build_metadata_similarity_graph(metadata, k_neighbors=3)

    for edge in result.edges:
        assert metadata[edge.song_id_a].genre_top == metadata[edge.song_id_b].genre_top


def test_build_metadata_similarity_graph_shared_album_can_cross_genres():
    """The whole point of strengthening the baseline beyond genre-only: a
    real non-audio signal (same album) can legitimately connect two
    different genres, unlike the old genre-only version where that was
    structurally impossible."""
    metadata = {
        1: SongMetadata(genre_top="Rock", album_id=99),
        2: SongMetadata(genre_top="Jazz", album_id=99),  # same album, different genre
        3: SongMetadata(genre_top="Pop", album_id=1),
        4: SongMetadata(genre_top="Classical", album_id=2),
    }

    result = build_metadata_similarity_graph(metadata, k_neighbors=1)

    edge_pairs = {frozenset((e.song_id_a, e.song_id_b)) for e in result.edges}
    assert frozenset((1, 2)) in edge_pairs


def test_build_metadata_similarity_graph_shared_tags_boost_similarity():
    metadata = {
        1: SongMetadata(genre_top="Rock", tags=frozenset({"chill", "instrumental"})),
        2: SongMetadata(genre_top="Rock", tags=frozenset({"chill", "instrumental"})),
        3: SongMetadata(genre_top="Rock", tags=frozenset({"aggressive"})),
    }

    result = build_metadata_similarity_graph(metadata, k_neighbors=1)

    edge_pairs = {frozenset((e.song_id_a, e.song_id_b)) for e in result.edges}
    assert frozenset((1, 2)) in edge_pairs  # closer match should win the single k=1 slot


def test_build_metadata_similarity_graph_empty_optional_signals_dont_crash():
    """No song has genres_all/tags data and no two share an album -- every
    optional signal matrix is all-zero, so this degrades to genre_sim/4
    everywhere, not a divide-by-zero or shape error."""
    metadata = {i: SongMetadata(genre_top="Rock" if i % 2 == 0 else "Jazz") for i in range(8)}

    result = build_metadata_similarity_graph(metadata, k_neighbors=2)

    assert len(result.nodes) == 8
    for edge in result.edges:
        assert 0.0 <= edge.weight <= 1.0 + 1e-9


def test_build_metadata_similarity_graph_clusters_still_match_genre():
    metadata = {i: SongMetadata(genre_top="Rock") for i in range(5)} | {
        i + 100: SongMetadata(genre_top="Jazz", album_id=1) for i in range(5)
    }

    result = build_metadata_similarity_graph(metadata, k_neighbors=2)

    rock_clusters = {n.cluster for n in result.nodes if metadata[n.song_id].genre_top == "Rock"}
    jazz_clusters = {n.cluster for n in result.nodes if metadata[n.song_id].genre_top == "Jazz"}
    assert len(rock_clusters) == 1
    assert len(jazz_clusters) == 1
    assert rock_clusters != jazz_clusters


def test_build_metadata_similarity_graph_edge_count_scales_with_k_not_genre_size():
    """Regression guard for the reason this reuses k-NN sampling instead of a
    literal per-genre clique: edge count must stay close to n*k/2, not blow
    up to O(songs_per_genre^2), even with a large single genre."""
    metadata = {i: SongMetadata(genre_top="Rock") for i in range(200)}

    result = build_metadata_similarity_graph(metadata, k_neighbors=4)

    assert len(result.edges) < 200 * 4  # well under a clique's ~19900 edges


def test_build_metadata_similarity_graph_all_nodes_present():
    genres = ["Rock", "Jazz", "Pop", "Rock", "Pop"]
    metadata = {i: SongMetadata(genre_top=g) for i, g in enumerate(genres)}

    result = build_metadata_similarity_graph(metadata, k_neighbors=2)

    assert {n.song_id for n in result.nodes} == set(metadata.keys())


def test_cross_genre_edge_fraction_no_edges_is_zero_not_undefined():
    assert cross_genre_edge_fraction([], {}) == 0.0


def test_cross_genre_edge_fraction_all_same_genre():
    genre_by_song = {1: "Rock", 2: "Rock", 3: "Rock"}
    edges = [GraphEdge(1, 2, 1.0), GraphEdge(2, 3, 1.0)]
    assert cross_genre_edge_fraction(edges, genre_by_song) == 0.0


def test_cross_genre_edge_fraction_mixed():
    genre_by_song = {1: "Rock", 2: "Jazz", 3: "Rock", 4: "Rock"}
    edges = [GraphEdge(1, 2, 1.0), GraphEdge(3, 4, 1.0)]  # one cross-genre, one not
    assert cross_genre_edge_fraction(edges, genre_by_song) == 0.5


def test_cosine_similarity_between_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity_between(v, v) == pytest.approx(1.0)


def test_cosine_similarity_between_orthogonal_vectors_is_zero():
    assert cosine_similarity_between(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_between_zero_vector_is_zero_not_nan():
    assert cosine_similarity_between(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


def test_pick_metadata_mismatch_pair_picks_lowest_real_similarity_edge():
    vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),  # identical to 1 -- high real similarity
        3: np.array([0.0, 1.0]),  # orthogonal to 1 -- low real similarity
    }
    metadata_edges = [GraphEdge(1, 2, 1.0), GraphEdge(1, 3, 1.0)]  # both "metadata-similar" (equal weight)

    pair = pick_metadata_mismatch_pair(metadata_edges, vectors)

    assert {pair.song_id_a, pair.song_id_b} == {1, 3}
    assert pair.audio_similarity == pytest.approx(0.0)


def test_pick_metadata_mismatch_pair_no_edges_returns_none():
    assert pick_metadata_mismatch_pair([], {1: np.array([1.0])}) is None


def test_pick_metadata_mismatch_pair_skips_edges_missing_vectors():
    vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    metadata_edges = [GraphEdge(1, 99, 1.0), GraphEdge(1, 2, 1.0)]  # song 99 has no vector

    pair = pick_metadata_mismatch_pair(metadata_edges, vectors)

    assert {pair.song_id_a, pair.song_id_b} == {1, 2}


def test_pick_real_cross_genre_pair_picks_strongest_cross_genre_edge():
    genre_by_song = {1: "Rock", 2: "Jazz", 3: "Rock", 4: "Rock"}
    real_edges = [
        GraphEdge(1, 2, 0.6),  # cross-genre, weaker
        GraphEdge(1, 4, 0.9),  # same-genre, stronger -- must be excluded
        GraphEdge(2, 3, 0.8),  # cross-genre, stronger
    ]

    pair = pick_real_cross_genre_pair(real_edges, genre_by_song)

    assert {pair.song_id_a, pair.song_id_b} == {2, 3}
    assert pair.audio_similarity == pytest.approx(0.8)


def test_pick_real_cross_genre_pair_no_cross_genre_edges_returns_none():
    genre_by_song = {1: "Rock", 2: "Rock"}
    real_edges = [GraphEdge(1, 2, 0.9)]

    assert pick_real_cross_genre_pair(real_edges, genre_by_song) is None


def test_compute_metadata_similarity_components_returns_four_matrices():
    metadata = {
        1: SongMetadata(genre_top="Rock", genres_all=frozenset({10}), album_id=1, tags=frozenset({"chill"})),
        2: SongMetadata(genre_top="Rock", genres_all=frozenset({10}), album_id=1, tags=frozenset({"chill"})),
        3: SongMetadata(genre_top="Jazz", genres_all=frozenset({20}), album_id=2, tags=frozenset({"loud"})),
    }

    song_ids, components = compute_metadata_similarity_components(metadata)

    assert song_ids == [1, 2, 3]
    assert set(components.keys()) == {"genre", "genres_all", "album", "tags"}
    for matrix in components.values():
        assert matrix.shape == (3, 3)
    # song 1 and 2 share everything -- every component should score them 1.0
    for name, matrix in components.items():
        assert matrix[0, 1] == pytest.approx(1.0), f"{name} should be 1.0 for identical songs 1 and 2"


def test_compute_metadata_similarity_components_fewer_than_two_songs_returns_empty():
    song_ids, components = compute_metadata_similarity_components({1: SongMetadata(genre_top="Rock")})
    assert song_ids == [1]
    assert components == {}


def test_combine_metadata_similarities_default_matches_default_metadata_weights():
    """Regression guard: the default must stay in sync with
    DEFAULT_METADATA_WEIGHTS -- the real, evaluated combination
    (notebooks/04_metadata_baseline_eda.ipynb), not silently drift back to an
    arbitrary equal-weight guess."""
    components = {
        "genre": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "genres_all": np.array([[0.0, 0.3], [0.3, 0.0]]),
        "album": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "tags": np.array([[0.0, 1.0], [1.0, 0.0]]),
    }

    combined = combine_metadata_similarities(components)
    expected = combine_metadata_similarities(components, weights=DEFAULT_METADATA_WEIGHTS)

    assert combined[0, 1] == pytest.approx(expected[0, 1])


def test_default_metadata_weights_is_not_equal_weighting():
    """DEFAULT_METADATA_WEIGHTS was chosen by a real evaluation (see
    notebooks/04_metadata_baseline_eda.ipynb) that found equal weighting is
    tied on genre-cohesion@10 with a tags/album-weighted variant, but the
    tags/album-weighted variant produces ~20x more genuine cross-genre
    edges -- this guards against silently reverting to the untested
    equal-weight guess."""
    values = set(DEFAULT_METADATA_WEIGHTS.values())
    assert len(values) > 1, "expected an intentionally uneven weighting, not equal weights"
    assert DEFAULT_METADATA_WEIGHTS["album"] > DEFAULT_METADATA_WEIGHTS["genre"]
    assert DEFAULT_METADATA_WEIGHTS["tags"] > DEFAULT_METADATA_WEIGHTS["genre"]


def test_combine_metadata_similarities_weights_need_not_sum_to_one():
    """Weights are normalized internally, so intuitive relative weights
    (not pre-normalized fractions) are safe to pass directly."""
    components = {
        "genre": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "tags": np.array([[0.0, 0.0], [0.0, 0.0]]),
    }

    combined = combine_metadata_similarities(components, weights={"genre": 3, "tags": 1})

    assert combined[0, 1] == pytest.approx(0.75)  # 3/(3+1)


def test_combine_metadata_similarities_missing_weight_defaults_to_zero_contribution():
    components = {
        "genre": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "tags": np.array([[0.0, 1.0], [1.0, 0.0]]),
    }

    combined = combine_metadata_similarities(components, weights={"genre": 1.0})  # tags omitted

    assert combined[0, 1] == pytest.approx(1.0)  # only genre contributes, tags weighted 0


def test_build_metadata_similarity_graph_custom_weights_change_edge_selection():
    """A non-default weighting must actually change which edges get picked
    -- confirms build_metadata_similarity_graph really uses the weights
    argument, not just accepting and ignoring it."""
    metadata = {
        1: SongMetadata(genre_top="Rock", album_id=None, tags=frozenset({"a", "b", "c"})),
        2: SongMetadata(genre_top="Jazz", album_id=None, tags=frozenset({"a", "b", "c"})),  # shares tags, not genre
        3: SongMetadata(genre_top="Rock", album_id=None, tags=frozenset({"x"})),  # shares genre, not tags
    }

    genre_heavy = build_metadata_similarity_graph(metadata, k_neighbors=1, weights={"genre": 1.0, "tags": 0.0})
    tags_heavy = build_metadata_similarity_graph(metadata, k_neighbors=1, weights={"genre": 0.0, "tags": 1.0})

    genre_heavy_pairs = {frozenset((e.song_id_a, e.song_id_b)) for e in genre_heavy.edges}
    tags_heavy_pairs = {frozenset((e.song_id_a, e.song_id_b)) for e in tags_heavy.edges}

    assert frozenset((1, 3)) in genre_heavy_pairs  # same genre wins when genre is all that's weighted
    assert frozenset((1, 2)) in tags_heavy_pairs  # same tags wins when tags is all that's weighted
