import numpy as np
import pytest

from sonic_explorer.models import Segment, Song
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository
from sonic_explorer.retrieval.song_level_index import build_song_level_index, query_song_level


@pytest.fixture
def conn():
    connection = init_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def repos(conn):
    return SongRepository(conn), EmbeddingRepository(conn)


def add_song_with_segments(song_repo, embedding_repo, track_id, vectors, facet_name="sound"):
    song = Song(
        filepath=f"/data/audio/{track_id}.mp3", fma_track_id=track_id, title=f"Song {track_id}",
        artist="Artist", genre_top="Rock", duration_sec=30.0,
    )
    song_id = song_repo.add_song(song)
    segments = [
        Segment(song_id=song_id, start_sec=i * 5.0, end_sec=(i + 1) * 5.0, segment_index=i)
        for i in range(len(vectors))
    ]
    seg_ids = song_repo.add_segments(song_id, segments)
    for seg_id, vec in zip(seg_ids, vectors):
        embedding_repo.add_vector(facet_name, seg_id, vec)
    return song_id


def test_build_song_level_index_returns_none_when_no_vectors(repos):
    song_repo, embedding_repo = repos
    assert build_song_level_index(song_repo, embedding_repo, "sound") is None


def test_query_song_level_finds_closest_song(repos):
    song_repo, embedding_repo = repos
    song_a = add_song_with_segments(
        song_repo, embedding_repo, 1,
        [np.array([1.0, 0.0], dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)],
    )
    song_b = add_song_with_segments(
        song_repo, embedding_repo, 2,
        [np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)],
    )

    index = build_song_level_index(song_repo, embedding_repo, "sound")
    results = query_song_level(index, np.array([0.9, 0.1], dtype=np.float32), k=2)

    assert results[0][0] == song_a
    assert results[1][0] == song_b
    assert results[0][1] > results[1][1]


def test_query_song_level_excludes_query_song(repos):
    song_repo, embedding_repo = repos
    song_a = add_song_with_segments(song_repo, embedding_repo, 1, [np.array([1.0, 0.0], dtype=np.float32)])
    song_b = add_song_with_segments(song_repo, embedding_repo, 2, [np.array([0.9, 0.1], dtype=np.float32)])

    index = build_song_level_index(song_repo, embedding_repo, "sound")
    results = query_song_level(index, np.array([1.0, 0.0], dtype=np.float32), k=5, exclude_song_id=song_a)

    assert [r[0] for r in results] == [song_b]


def test_query_song_level_averages_segments_per_song():
    """A song's mean-pooled vector should reflect all its segments, not just
    the first one -- this is the whole point of song-level aggregation."""
    conn = init_db(":memory:")
    song_repo, embedding_repo = SongRepository(conn), EmbeddingRepository(conn)
    song_id = add_song_with_segments(
        song_repo, embedding_repo, 1,
        [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)],
    )

    index = build_song_level_index(song_repo, embedding_repo, "sound")
    # querying with the average direction should score higher than either single-segment direction alone
    results_avg = query_song_level(index, np.array([1.0, 1.0], dtype=np.float32), k=1)
    assert results_avg[0][0] == song_id
    assert results_avg[0][1] > 0.9  # near-perfect match to the averaged [1,1]-normalized direction
    conn.close()
