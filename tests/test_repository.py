import numpy as np
import pytest

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
def song_repo(conn):
    return SongRepository(conn)


@pytest.fixture
def embedding_repo(conn):
    return EmbeddingRepository(conn)


def make_song(track_id=1, title="Test Song"):
    return Song(
        filepath=f"/data/audio/{track_id}.mp3",
        fma_track_id=track_id,
        title=title,
        artist="Test Artist",
        genre_top="Rock",
        duration_sec=180.0,
    )


def test_add_song_returns_id(song_repo):
    song_id = song_repo.add_song(make_song())
    assert song_id == 1


def test_add_song_is_idempotent_on_fma_track_id(song_repo):
    first_id = song_repo.add_song(make_song(track_id=42))
    second_id = song_repo.add_song(make_song(track_id=42, title="Different title, same track"))
    assert first_id == second_id


def test_get_song_roundtrip(song_repo):
    song_id = song_repo.add_song(make_song(track_id=7))
    fetched = song_repo.get_song(song_id)
    assert fetched.fma_track_id == 7
    assert fetched.title == "Test Song"
    assert fetched.segments == []


def test_get_song_missing_returns_none(song_repo):
    assert song_repo.get_song(999) is None


def test_list_songs_filters_by_genre(song_repo):
    song_repo.add_song(make_song(track_id=1))
    jazz = Song(
        filepath="/data/audio/2.mp3", fma_track_id=2, title="Jazz Song",
        artist="A", genre_top="Jazz", duration_sec=200.0,
    )
    song_repo.add_song(jazz)

    rock_songs = song_repo.list_songs(genre="Rock")
    assert len(rock_songs) == 1
    assert rock_songs[0].genre_top == "Rock"
    assert len(song_repo.list_songs()) == 2


def test_add_segments_and_get_segments(song_repo):
    song_id = song_repo.add_song(make_song())
    segments = [
        Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0),
        Segment(song_id=song_id, start_sec=2.5, end_sec=7.5, segment_index=1),
    ]
    ids = song_repo.add_segments(song_id, segments)
    assert len(ids) == 2

    fetched = song_repo.get_segments(song_id)
    assert len(fetched) == 2
    assert fetched[0].segment_index == 0


def test_add_segments_is_idempotent(song_repo):
    song_id = song_repo.add_song(make_song())
    segments = [Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0)]
    first_ids = song_repo.add_segments(song_id, segments)
    second_ids = song_repo.add_segments(song_id, segments)
    assert first_ids == second_ids
    assert len(song_repo.get_segments(song_id)) == 1


def test_embedding_status_defaults_to_pending(embedding_repo, song_repo):
    song_id = song_repo.add_song(make_song())
    [seg_id] = song_repo.add_segments(
        song_id, [Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0)]
    )
    assert embedding_repo.status(seg_id, "sound") == "pending"


def test_add_vector_marks_done_and_is_searchable(embedding_repo, song_repo):
    song_id = song_repo.add_song(make_song())
    [seg_id] = song_repo.add_segments(
        song_id, [Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0)]
    )
    vector = np.random.rand(8).astype(np.float32)
    embedding_repo.add_vector("sound", seg_id, vector)

    assert embedding_repo.status(seg_id, "sound") == "done"
    assert embedding_repo.index_size("sound") == 1

    results = embedding_repo.search("sound", vector, k=5)
    assert results[0][0] == seg_id
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_search_ranks_closest_vector_first(embedding_repo, song_repo):
    song_id = song_repo.add_song(make_song())
    seg_ids = song_repo.add_segments(
        song_id,
        [
            Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0),
            Segment(song_id=song_id, start_sec=5.0, end_sec=10.0, segment_index=1),
        ],
    )
    embedding_repo.add_vector("sound", seg_ids[0], np.array([1.0, 0.0], dtype=np.float32))
    embedding_repo.add_vector("sound", seg_ids[1], np.array([0.0, 1.0], dtype=np.float32))

    results = embedding_repo.search("sound", np.array([0.9, 0.1], dtype=np.float32), k=2)
    assert results[0][0] == seg_ids[0]
    assert results[1][0] == seg_ids[1]


def test_search_on_empty_index_returns_empty_list(embedding_repo):
    results = embedding_repo.search("sound", np.array([1.0, 0.0], dtype=np.float32), k=5)
    assert results == []
