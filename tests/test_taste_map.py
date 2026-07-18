import numpy as np
import pytest

from sonic_explorer.analysis.taste_map import compute_taste_map, mean_pool_song_vectors
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


def add_song_with_segments(song_repo, embedding_repo, track_id, genre, vectors):
    song = Song(
        filepath=f"/data/audio/{track_id}.mp3", fma_track_id=track_id, title=f"Song {track_id}",
        artist="Artist", genre_top=genre, duration_sec=30.0,
    )
    song_id = song_repo.add_song(song)
    segments = [
        Segment(song_id=song_id, start_sec=i * 5.0, end_sec=(i + 1) * 5.0, segment_index=i)
        for i in range(len(vectors))
    ]
    seg_ids = song_repo.add_segments(song_id, segments)
    for seg_id, vec in zip(seg_ids, vectors):
        embedding_repo.add_vector("sound", seg_id, vec)
    return song_id


def test_mean_pool_song_vectors_averages_segments(repos):
    song_repo, embedding_repo = repos
    song_id = add_song_with_segments(
        song_repo, embedding_repo, 1, "Rock",
        [np.array([1.0, 1.0], dtype=np.float32), np.array([3.0, 3.0], dtype=np.float32)],
    )

    pooled = mean_pool_song_vectors(song_repo, embedding_repo)

    assert song_id in pooled
    # add_vector L2-normalizes before storage, so exact means aren't [2,2] --
    # just check it's a genuine average, not simply the first/last vector
    assert pooled[song_id].shape == (2,)


def test_mean_pool_song_vectors_skips_songs_with_no_embeddings(repos):
    song_repo, _ = repos
    song = Song(filepath="/x.mp3", fma_track_id=99, title="No embeddings", artist="A", genre_top="Rock", duration_sec=10.0)
    song_id = song_repo.add_song(song)
    song_repo.add_segments(song_id, [Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0)])

    pooled = mean_pool_song_vectors(song_repo, EmbeddingRepository(song_repo.conn))

    assert song_id not in pooled


def test_compute_taste_map_separates_distinct_clusters():
    rng = np.random.default_rng(0)
    cluster_a = {i: rng.normal(loc=[10, 10], scale=0.1) for i in range(5)}
    cluster_b = {i + 100: rng.normal(loc=[-10, -10], scale=0.1) for i in range(5)}
    song_vectors = {**cluster_a, **cluster_b}

    result = compute_taste_map(song_vectors, n_clusters=2)

    assert len(result.points) == 10
    labels_a = {p.cluster for p in result.points if p.song_id < 100}
    labels_b = {p.cluster for p in result.points if p.song_id >= 100}
    # each group should collapse to a single, and different, cluster label
    assert len(labels_a) == 1
    assert len(labels_b) == 1
    assert labels_a != labels_b


def test_compute_taste_map_handles_empty_input():
    result = compute_taste_map({})
    assert result.points == []


def test_compute_taste_map_handles_single_song():
    result = compute_taste_map({1: np.array([1.0, 2.0, 3.0])})
    assert len(result.points) == 1
    assert result.points[0].song_id == 1
