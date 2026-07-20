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


def test_update_filepath(song_repo):
    song_id = song_repo.add_song(make_song(track_id=7))
    song_repo.update_filepath(song_id, "/data/audio/7.mp3")
    assert song_repo.get_song(song_id).filepath == "/data/audio/7.mp3"


def test_new_song_has_no_song_dna_by_default(song_repo):
    song_id = song_repo.add_song(make_song(track_id=7))
    song = song_repo.get_song(song_id)
    assert song.tempo_bpm is None
    assert song.energy is None


def test_update_song_dna_round_trips(song_repo):
    song_id = song_repo.add_song(make_song(track_id=7))
    song_repo.update_song_dna(
        song_id, tempo_bpm=120.0, energy=0.3, brightness=2000.0,
        harmonic_complexity=0.6, rhythmic_density=2.5,
    )
    song = song_repo.get_song(song_id)
    assert song.tempo_bpm == 120.0
    assert song.energy == 0.3
    assert song.brightness == 2000.0
    assert song.harmonic_complexity == 0.6
    assert song.rhythmic_density == 2.5


def test_migration_adds_song_dna_columns_to_pre_existing_db(tmp_path):
    """Simulates a DB created before song DNA existed (just the base schema, no
    migration run) -- init_db() on it must add the new columns without losing
    existing data, not just work on brand-new DBs."""
    import sqlite3

    from sonic_explorer.repository.db import SCHEMA, init_db

    db_path = tmp_path / "old.db"
    raw_conn = sqlite3.connect(str(db_path))
    raw_conn.executescript(SCHEMA)  # base schema only, no migration
    raw_conn.execute(
        "INSERT INTO songs (fma_track_id, filepath, title, artist, genre_top, duration_sec) "
        "VALUES (1, '/x.mp3', 'Old Song', 'Artist', 'Rock', 30.0)"
    )
    raw_conn.commit()
    raw_conn.close()

    conn = init_db(db_path)  # should migrate in place
    repo = SongRepository(conn)
    song = repo.get_song_by_fma_track_id(1)

    assert song.title == "Old Song"  # pre-existing data preserved
    assert song.tempo_bpm is None  # new column present, just empty
    repo.update_song_dna(song.id, 100.0, 0.1, 1500.0, 0.5, 1.0)
    assert repo.get_song(song.id).tempo_bpm == 100.0


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


def test_new_song_is_not_saved_by_default(song_repo):
    song_id = song_repo.add_song(make_song(track_id=7))
    assert song_repo.get_song(song_id).is_saved is False


def test_save_and_unsave_song(song_repo):
    song_id = song_repo.add_song(make_song(track_id=7))

    song_repo.save_song(song_id)
    assert song_repo.get_song(song_id).is_saved is True

    song_repo.unsave_song(song_id)
    assert song_repo.get_song(song_id).is_saved is False


def test_list_songs_saved_only_filter(song_repo):
    saved_id = song_repo.add_song(make_song(track_id=1, title="Saved Song"))
    song_repo.add_song(make_song(track_id=2, title="Unsaved Song"))
    song_repo.save_song(saved_id)

    saved = song_repo.list_songs(saved_only=True)
    assert len(saved) == 1
    assert saved[0].id == saved_id
    assert len(song_repo.list_songs()) == 2


def test_list_songs_saved_only_combines_with_genre_filter(song_repo):
    rock_saved = song_repo.add_song(make_song(track_id=1, title="Rock Saved"))
    song_repo.add_song(Song(filepath="/x.mp3", fma_track_id=2, title="Jazz Saved", artist="A",
                             genre_top="Jazz", duration_sec=100.0))
    song_repo.save_song(rock_saved)
    jazz_id = song_repo.get_song_by_fma_track_id(2).id
    song_repo.save_song(jazz_id)

    rock_saved_songs = song_repo.list_songs(genre="Rock", saved_only=True)
    assert len(rock_saved_songs) == 1
    assert rock_saved_songs[0].title == "Rock Saved"


def test_migration_adds_is_saved_column_to_pre_existing_db(tmp_path):
    """Same class of regression as the song-DNA migration test: a DB created
    before is_saved existed must gain the column (defaulting to unsaved)
    without losing existing data."""
    import sqlite3

    from sonic_explorer.repository.db import SCHEMA, init_db

    db_path = tmp_path / "old.db"
    raw_conn = sqlite3.connect(str(db_path))
    raw_conn.executescript(SCHEMA)
    raw_conn.execute(
        "INSERT INTO songs (fma_track_id, filepath, title, artist, genre_top, duration_sec) "
        "VALUES (1, '/x.mp3', 'Old Song', 'Artist', 'Rock', 30.0)"
    )
    raw_conn.commit()
    raw_conn.close()

    conn = init_db(db_path)
    repo = SongRepository(conn)
    song = repo.get_song_by_fma_track_id(1)

    assert song.title == "Old Song"
    assert song.is_saved is False
    repo.save_song(song.id)
    assert repo.get_song(song.id).is_saved is True


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


def test_mark_skipped_sets_status_without_a_vector(embedding_repo, song_repo):
    song_id = song_repo.add_song(make_song())
    [seg_id] = song_repo.add_segments(
        song_id, [Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0)]
    )

    embedding_repo.mark_skipped(seg_id, "vocal")

    assert embedding_repo.status(seg_id, "vocal") == "skipped"
    assert embedding_repo.index_size("vocal") == 0  # no vector was ever added


def test_mark_skipped_is_idempotent(embedding_repo, song_repo):
    song_id = song_repo.add_song(make_song())
    [seg_id] = song_repo.add_segments(
        song_id, [Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0)]
    )

    embedding_repo.mark_skipped(seg_id, "vocal")
    embedding_repo.mark_skipped(seg_id, "vocal")  # must not raise (UNIQUE(segment_id, facet_name))

    assert embedding_repo.status(seg_id, "vocal") == "skipped"


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


def test_get_structure_matrix_round_trips(conn, tmp_path):
    repo = EmbeddingRepository(conn, artifacts_dir=tmp_path)
    (tmp_path / "structure").mkdir()
    matrix = np.random.default_rng(0).random((10, 10)).astype(np.float32)
    np.save(tmp_path / "structure" / "5.npy", matrix)

    loaded = repo.get_structure_matrix(5)
    assert np.array_equal(loaded, matrix)


def test_get_structure_matrix_missing_raises(conn, tmp_path):
    # uses an isolated tmp artifacts_dir, not the shared embedding_repo fixture's
    # default (config.ARTIFACTS_DIR) -- that points at this project's real data/
    # artifacts/, where song_id 999 genuinely exists among the 1400 real songs
    repo = EmbeddingRepository(conn, artifacts_dir=tmp_path)
    (tmp_path / "structure").mkdir()
    with pytest.raises(FileNotFoundError):
        repo.get_structure_matrix(999)


def test_get_structure_timeline_round_trips_including_fingerprint(conn, tmp_path):
    repo = EmbeddingRepository(conn, artifacts_dir=tmp_path)
    (tmp_path / "structure").mkdir()
    fp = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    np.savez(
        tmp_path / "structure" / "5_timeline.npz",
        starts=np.array([0.0, 5.0], dtype=np.float32),
        ends=np.array([5.0, 10.0], dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int32),
        sound_fp=fp,
    )

    timeline = repo.get_structure_timeline(5)
    assert list(timeline.segment_starts) == [0.0, 5.0]
    assert list(timeline.segment_ends) == [5.0, 10.0]
    assert list(timeline.segment_labels) == [0, 1]
    assert np.array_equal(timeline.sound_fingerprint, fp)


def test_get_structure_timeline_missing_fingerprint_key_is_none(conn, tmp_path):
    """Backward compat: timeline files written before sound fingerprints existed."""
    repo = EmbeddingRepository(conn, artifacts_dir=tmp_path)
    (tmp_path / "structure").mkdir()
    np.savez(
        tmp_path / "structure" / "5_timeline.npz",
        starts=np.array([0.0], dtype=np.float32),
        ends=np.array([10.0], dtype=np.float32),
        labels=np.array([0], dtype=np.int32),
    )

    timeline = repo.get_structure_timeline(5)
    assert timeline.sound_fingerprint is None


def test_get_structure_timeline_round_trips_harmony_fingerprint(conn, tmp_path):
    repo = EmbeddingRepository(conn, artifacts_dir=tmp_path)
    (tmp_path / "structure").mkdir()
    harmony_fp = np.random.default_rng(0).random((12, 32)).astype(np.float32)
    np.savez(
        tmp_path / "structure" / "5_timeline.npz",
        starts=np.array([0.0], dtype=np.float32),
        ends=np.array([10.0], dtype=np.float32),
        labels=np.array([0], dtype=np.int32),
        harmony_fp=harmony_fp,
    )

    timeline = repo.get_structure_timeline(5)
    assert np.array_equal(timeline.harmony_fingerprint, harmony_fp)


def test_get_structure_timeline_missing_harmony_fingerprint_key_is_none(conn, tmp_path):
    """Backward compat: timeline files written before harmony fingerprints existed."""
    repo = EmbeddingRepository(conn, artifacts_dir=tmp_path)
    (tmp_path / "structure").mkdir()
    np.savez(
        tmp_path / "structure" / "5_timeline.npz",
        starts=np.array([0.0], dtype=np.float32),
        ends=np.array([10.0], dtype=np.float32),
        labels=np.array([0], dtype=np.int32),
    )

    timeline = repo.get_structure_timeline(5)
    assert timeline.harmony_fingerprint is None


def test_get_structure_timeline_round_trips_novelty_fields(conn, tmp_path):
    repo = EmbeddingRepository(conn, artifacts_dir=tmp_path)
    (tmp_path / "structure").mkdir()
    novelty = np.array([0.0, 0.2, 0.9, 0.1], dtype=np.float32)
    novelty_times = np.array([0.0, 2.5, 5.0, 7.5], dtype=np.float32)
    np.savez(
        tmp_path / "structure" / "5_timeline.npz",
        starts=np.array([0.0], dtype=np.float32),
        ends=np.array([10.0], dtype=np.float32),
        labels=np.array([0], dtype=np.int32),
        novelty=novelty,
        novelty_times=novelty_times,
        has_clear_structure=True,
        structural_confidence=0.42,
    )

    timeline = repo.get_structure_timeline(5)
    assert np.array_equal(timeline.novelty_curve, novelty)
    assert np.array_equal(timeline.novelty_times, novelty_times)
    assert timeline.has_clear_structure is True
    assert timeline.structural_confidence == pytest.approx(0.42)


def test_get_structure_timeline_missing_novelty_keys_defaults_to_clear_structure(conn, tmp_path):
    """Backward compat: pre-novelty timeline files default has_clear_structure to
    True -- preserves the old behavior (always show segments) rather than
    silently hiding every already-synced song's timeline."""
    repo = EmbeddingRepository(conn, artifacts_dir=tmp_path)
    (tmp_path / "structure").mkdir()
    np.savez(
        tmp_path / "structure" / "5_timeline.npz",
        starts=np.array([0.0], dtype=np.float32),
        ends=np.array([10.0], dtype=np.float32),
        labels=np.array([0], dtype=np.int32),
    )

    timeline = repo.get_structure_timeline(5)
    assert timeline.novelty_curve is None
    assert timeline.has_clear_structure is True
    assert timeline.structural_confidence is None
