import numpy as np

from sonic_explorer.evaluation.genre_free_clustering import cluster_and_compare_to_genre


def _make_genre_separated_vectors(n_per_genre=15, n_genres=8, dim=32, seed=1):
    """8 tight, well-separated Gaussian blobs, one per genre -- a case where
    audio clusters SHOULD recover genre almost perfectly, so ARI must land
    near 1.0. The real test of correctness: does this function actually
    measure agreement, not just return some fixed number regardless of
    input."""
    rng = np.random.default_rng(seed)
    song_vectors, genre_by_song = {}, {}
    song_id = 0
    for g in range(n_genres):
        center = rng.normal(scale=20.0, size=dim)  # far-apart centers relative to the tiny per-cluster spread below
        for _ in range(n_per_genre):
            song_vectors[song_id] = center + rng.normal(scale=0.1, size=dim)
            genre_by_song[song_id] = f"genre_{g}"
            song_id += 1
    return song_vectors, genre_by_song


def test_ari_near_one_when_audio_clusters_perfectly_match_genre():
    song_vectors, genre_by_song = _make_genre_separated_vectors()

    result = cluster_and_compare_to_genre(song_vectors, genre_by_song, n_clusters=8, seed=42)

    assert result.adjusted_rand_index > 0.9
    assert result.n_songs == len(song_vectors)
    assert sum(result.cluster_sizes) == result.n_songs
    assert sum(result.genre_sizes.values()) == result.n_songs


def test_ari_near_zero_when_genre_labels_are_shuffled_random_noise():
    """Same well-separated audio clusters as above, but genre labels
    reassigned uniformly at random -- now genre carries no real
    relationship to the audio clusters, so ARI must land near 0, not near
    1.0. Confirms the function is actually sensitive to real agreement,
    not just returning a high number whenever clusters happen to exist."""
    song_vectors, _ = _make_genre_separated_vectors()
    rng = np.random.default_rng(7)
    random_genres = {sid: f"genre_{rng.integers(0, 8)}" for sid in song_vectors}

    result = cluster_and_compare_to_genre(song_vectors, random_genres, n_clusters=8, seed=42)

    assert result.adjusted_rand_index < 0.15


def test_songs_missing_a_genre_label_are_excluded():
    song_vectors, genre_by_song = _make_genre_separated_vectors(n_per_genre=5)
    incomplete_genres = dict(list(genre_by_song.items())[:-3])  # drop genre labels for the last 3 songs

    result = cluster_and_compare_to_genre(song_vectors, incomplete_genres, n_clusters=8, seed=42)

    assert result.n_songs == len(song_vectors) - 3
