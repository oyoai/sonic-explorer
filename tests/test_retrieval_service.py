import numpy as np
import pytest

from sonic_explorer.models import Segment, Song
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository
from sonic_explorer.retrieval.service import RetrievalService


@pytest.fixture
def conn():
    connection = init_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def repos(conn):
    return SongRepository(conn), EmbeddingRepository(conn)


def add_song_with_segments(song_repo, embedding_repo, track_id, title, genre, vectors):
    song = Song(
        filepath=f"/data/audio/{track_id}.mp3", fma_track_id=track_id, title=title,
        artist="Artist", genre_top=genre, duration_sec=30.0,
    )
    song_id = song_repo.add_song(song)
    segments = [
        Segment(song_id=song_id, start_sec=i * 5.0, end_sec=(i + 1) * 5.0, segment_index=i)
        for i in range(len(vectors))
    ]
    seg_ids = song_repo.add_segments(song_id, segments)
    for seg_id, vec in zip(seg_ids, vectors, strict=False):
        embedding_repo.add_vector("sound", seg_id, vec)
    return song_id, seg_ids


def test_query_by_segment_excludes_same_song_by_default(repos):
    song_repo, embedding_repo = repos
    service = RetrievalService(song_repo, embedding_repo)

    song_a_id, seg_a_ids = add_song_with_segments(
        song_repo, embedding_repo, 1, "Song A", "Rock",
        [np.array([1.0, 0.0], dtype=np.float32), np.array([0.9, 0.1], dtype=np.float32)],
    )
    song_b_id, seg_b_ids = add_song_with_segments(
        song_repo, embedding_repo, 2, "Song B", "Rock",
        [np.array([0.95, 0.05], dtype=np.float32)],
    )

    matches = service.query_by_segment(seg_a_ids[0], k=5)

    assert all(m.song.id != song_a_id for m in matches)
    assert any(m.song.id == song_b_id for m in matches)


def test_query_by_segment_can_include_same_song(repos):
    song_repo, embedding_repo = repos
    service = RetrievalService(song_repo, embedding_repo)

    song_a_id, seg_a_ids = add_song_with_segments(
        song_repo, embedding_repo, 1, "Song A", "Rock",
        [np.array([1.0, 0.0], dtype=np.float32), np.array([0.9, 0.1], dtype=np.float32)],
    )

    matches = service.query_by_segment(seg_a_ids[0], k=5, exclude_same_song=False)

    assert any(m.segment.id == seg_a_ids[1] for m in matches)
    assert all(m.segment.id != seg_a_ids[0] for m in matches)  # the query segment itself is always excluded


def test_query_by_segment_ranks_by_similarity(repos):
    song_repo, embedding_repo = repos
    service = RetrievalService(song_repo, embedding_repo)

    query_song_id, query_seg_ids = add_song_with_segments(
        song_repo, embedding_repo, 1, "Query", "Rock", [np.array([1.0, 0.0], dtype=np.float32)]
    )
    close_id, close_seg_ids = add_song_with_segments(
        song_repo, embedding_repo, 2, "Close", "Rock", [np.array([0.99, 0.05], dtype=np.float32)]
    )
    far_id, far_seg_ids = add_song_with_segments(
        song_repo, embedding_repo, 3, "Far", "Jazz", [np.array([0.0, 1.0], dtype=np.float32)]
    )

    matches = service.query_by_segment(query_seg_ids[0], k=5)

    assert matches[0].song.id == close_id
    assert matches[-1].song.id == far_id
    assert matches[0].score > matches[-1].score


def test_query_by_vector_returns_matches_directly(repos):
    song_repo, embedding_repo = repos
    service = RetrievalService(song_repo, embedding_repo)

    song_id, seg_ids = add_song_with_segments(
        song_repo, embedding_repo, 1, "Song A", "Rock", [np.array([1.0, 0.0], dtype=np.float32)]
    )

    matches = service.query_by_vector(np.array([1.0, 0.0], dtype=np.float32), k=5)

    assert len(matches) == 1
    assert matches[0].song.id == song_id
    assert matches[0].score == pytest.approx(1.0, abs=1e-4)
