import numpy as np
import pytest

from sonic_explorer.evaluation.genre_cohesion import (
    genre_cohesion_at_k,
    metadata_genre_cohesion_at_k,
    song_level_genre_cohesion_at_k,
)
from sonic_explorer.models import Segment, Song
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository


@pytest.fixture
def conn():
    connection = init_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def repos(conn):
    return SongRepository(conn), EmbeddingRepository(conn)


def add_song(song_repo, embedding_repo, track_id, genre, vector, facet_name="sound"):
    song = Song(
        filepath=f"/data/audio/{track_id}.mp3", fma_track_id=track_id, title=f"Song {track_id}",
        artist="Artist", genre_top=genre, duration_sec=10.0,
    )
    song_id = song_repo.add_song(song)
    [seg_id] = song_repo.add_segments(song_id, [Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0)])
    embedding_repo.add_vector(facet_name, seg_id, vector)
    return song_id


def test_genre_cohesion_higher_than_random_for_genre_clustered_data(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(1)

    # two well-separated genre clusters, several songs each
    for i in range(6):
        vec = rng.normal(loc=[10, 10], scale=0.2, size=2).astype(np.float32)
        add_song(song_repo, embedding_repo, i, "Rock", vec)
    for i in range(6, 12):
        vec = rng.normal(loc=[-10, -10], scale=0.2, size=2).astype(np.float32)
        add_song(song_repo, embedding_repo, i, "Jazz", vec)

    result = genre_cohesion_at_k(song_repo, embedding_repo, facet_name="sound", k=3)

    assert result.n_queries == 12
    assert result.observed > result.random_baseline
    assert result.observed == pytest.approx(1.0, abs=1e-6)  # clusters are tight and well-separated


def test_genre_cohesion_excludes_query_song_itself(repos):
    song_repo, embedding_repo = repos
    # a single song -- its only "neighbor" would be itself, so with no other songs
    # there should be no valid neighbors at all
    add_song(song_repo, embedding_repo, 1, "Rock", np.array([1.0, 0.0], dtype=np.float32))

    result = genre_cohesion_at_k(song_repo, embedding_repo, facet_name="sound", k=3)

    assert result.n_queries == 1
    assert result.observed == 0.0
    assert result.random_baseline == 0.0


def test_genre_cohesion_handles_no_embedded_segments(repos):
    song_repo, embedding_repo = repos
    result = genre_cohesion_at_k(song_repo, embedding_repo, facet_name="sound", k=3)
    assert result.n_queries == 0
    assert result.observed == 0.0
    assert result.random_baseline == 0.0


def test_genre_cohesion_respects_sample_size(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(2)
    for i in range(20):
        vec = rng.normal(size=2).astype(np.float32)
        add_song(song_repo, embedding_repo, i, "Rock" if i % 2 == 0 else "Jazz", vec)

    result = genre_cohesion_at_k(song_repo, embedding_repo, facet_name="sound", k=3, sample_size=5)
    assert result.n_queries == 5


def test_song_level_genre_cohesion_higher_than_random_for_genre_clustered_data(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(1)

    for i in range(6):
        vec = rng.normal(loc=[10, 10], scale=0.2, size=2).astype(np.float32)
        add_song(song_repo, embedding_repo, i, "Rock", vec)
    for i in range(6, 12):
        vec = rng.normal(loc=[-10, -10], scale=0.2, size=2).astype(np.float32)
        add_song(song_repo, embedding_repo, i, "Jazz", vec)

    result = song_level_genre_cohesion_at_k(song_repo, embedding_repo, facet_name="sound", k=3)

    assert result.n_queries == 12
    assert result.observed > result.random_baseline
    assert result.observed == pytest.approx(1.0, abs=1e-6)


def test_song_level_genre_cohesion_handles_no_embedded_segments(repos):
    song_repo, embedding_repo = repos
    result = song_level_genre_cohesion_at_k(song_repo, embedding_repo, facet_name="sound", k=3)
    assert result.n_queries == 0
    assert result.observed == 0.0
    assert result.random_baseline == 0.0


def test_metadata_genre_cohesion_perfect_similarity_within_genre_scores_1():
    """Two perfectly-separated genre blocks (similarity 1.0 within a genre,
    0.0 across) must score observed=1.0 -- every top-k neighbor shares
    genre, by construction of the similarity matrix itself."""
    song_ids = list(range(12))
    genre_by_song = {sid: ("Rock" if sid < 6 else "Jazz") for sid in song_ids}
    sims = np.zeros((12, 12))
    sims[:6, :6] = 1.0
    sims[6:, 6:] = 1.0

    result = metadata_genre_cohesion_at_k(song_ids, sims, genre_by_song, k=3)

    assert result.n_queries == 12
    assert result.observed == pytest.approx(1.0)
    assert result.observed > result.random_baseline


def test_metadata_genre_cohesion_excludes_query_song_itself():
    """A single song has no valid neighbors -- must not crash or count
    itself as its own nearest neighbor."""
    song_ids = [1]
    genre_by_song = {1: "Rock"}
    sims = np.zeros((1, 1))

    result = metadata_genre_cohesion_at_k(song_ids, sims, genre_by_song, k=3)

    assert result.n_queries == 0
    assert result.observed == 0.0
    assert result.random_baseline == 0.0


def test_metadata_genre_cohesion_respects_sample_size():
    song_ids = list(range(20))
    genre_by_song = {sid: ("Rock" if sid % 2 == 0 else "Jazz") for sid in song_ids}
    sims = np.random.default_rng(3).uniform(size=(20, 20))

    result = metadata_genre_cohesion_at_k(song_ids, sims, genre_by_song, k=3, sample_size=5, seed=1)

    assert result.n_queries == 5


def test_metadata_genre_cohesion_k_larger_than_available_songs_is_clamped():
    song_ids = list(range(4))
    genre_by_song = {sid: "Rock" for sid in song_ids}
    sims = np.ones((4, 4))

    result = metadata_genre_cohesion_at_k(song_ids, sims, genre_by_song, k=50)

    assert result.n_queries == 4
    assert result.observed == pytest.approx(1.0)


def test_metadata_genre_cohesion_label_becomes_facet_name():
    song_ids = [1, 2]
    genre_by_song = {1: "Rock", 2: "Jazz"}
    sims = np.array([[0.0, 0.5], [0.5, 0.0]])

    result = metadata_genre_cohesion_at_k(song_ids, sims, genre_by_song, k=1, label="genre_weighted_variant")

    assert result.facet_name == "genre_weighted_variant"
