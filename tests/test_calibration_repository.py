import pytest

from sonic_explorer.models import Segment, Song
from sonic_explorer.repository.calibration_repository import CalibrationRepository
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.song_repository import SongRepository


@pytest.fixture
def conn():
    connection = init_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def three_segments(conn):
    song_repo = SongRepository(conn)
    song = Song(filepath="/a.mp3", fma_track_id=1, title="A", artist="Artist", genre_top="Rock", duration_sec=30.0)
    song_id = song_repo.add_song(song)
    seg_ids = song_repo.add_segments(
        song_id,
        [
            Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0),
            Segment(song_id=song_id, start_sec=5.0, end_sec=10.0, segment_index=1),
            Segment(song_id=song_id, start_sec=10.0, end_sec=15.0, segment_index=2),
        ],
    )
    return seg_ids


def test_add_choice_and_count(conn, three_segments):
    repo = CalibrationRepository(conn)
    seg_x, seg_a, seg_b = three_segments

    repo.add_choice(seg_x, seg_a, seg_b, choice="a", rater="offi")

    assert repo.count() == 1
    ratings = repo.get_all_ratings()
    assert ratings[0]["segment_x_id"] == seg_x
    assert ratings[0]["segment_a_id"] == seg_a
    assert ratings[0]["segment_b_id"] == seg_b
    assert ratings[0]["choice"] == "a"
    assert ratings[0]["rater"] == "offi"


def test_add_choice_rejects_invalid_choice(conn, three_segments):
    repo = CalibrationRepository(conn)
    seg_x, seg_a, seg_b = three_segments

    with pytest.raises(ValueError):
        repo.add_choice(seg_x, seg_a, seg_b, choice="c")


def test_rated_triplet_keys_normalizes_ab_order(conn, three_segments):
    """The (x, a, b) key must not care which candidate landed in slot A vs
    B -- the same underlying triplet, shown with A/B swapped, is still the
    same triplet for dedup purposes."""
    repo = CalibrationRepository(conn)
    seg_x, seg_a, seg_b = three_segments

    repo.add_choice(seg_x, seg_a, seg_b, choice="b")

    keys = repo.rated_triplet_keys()
    assert (seg_x, min(seg_a, seg_b), max(seg_a, seg_b)) in keys


def test_count_with_no_ratings_is_zero(conn):
    repo = CalibrationRepository(conn)
    assert repo.count() == 0
    assert repo.get_all_ratings() == []
    assert repo.rated_triplet_keys() == set()


def test_add_choice_without_rater_is_optional(conn, three_segments):
    repo = CalibrationRepository(conn)
    seg_x, seg_a, seg_b = three_segments

    repo.add_choice(seg_x, seg_a, seg_b, choice="a")

    assert repo.get_all_ratings()[0]["rater"] is None


def _create_old_shape_calibration_table(db_path):
    """Simulates a DB created before the XAB migration -- the pair-rating
    shape (segment_a_id, segment_b_id, rating), no segment_x_id/choice."""
    import sqlite3

    raw_conn = sqlite3.connect(str(db_path))
    raw_conn.executescript("""
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fma_track_id INTEGER UNIQUE NOT NULL,
            filepath TEXT NOT NULL, title TEXT, artist TEXT, genre_top TEXT, duration_sec REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            song_id INTEGER NOT NULL REFERENCES songs(id),
            start_sec REAL NOT NULL, end_sec REAL NOT NULL, segment_index INTEGER NOT NULL,
            UNIQUE(song_id, start_sec, end_sec)
        );
        CREATE TABLE IF NOT EXISTS calibration_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_a_id INTEGER NOT NULL REFERENCES segments(id),
            segment_b_id INTEGER NOT NULL REFERENCES segments(id),
            rating REAL NOT NULL,
            rater TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    raw_conn.commit()
    return raw_conn


def test_migration_replaces_empty_old_schema_calibration_table(tmp_path):
    """A DB created before the XAB migration, with zero calibration rows,
    must have calibration_ratings silently replaced with the new
    (segment_x_id, segment_a_id, segment_b_id, choice) shape."""
    db_path = tmp_path / "old.db"
    raw_conn = _create_old_shape_calibration_table(db_path)
    raw_conn.close()

    conn = init_db(db_path)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(calibration_ratings)")}
    assert cols == {"id", "segment_x_id", "segment_a_id", "segment_b_id", "choice", "rater", "created_at"}

    repo = CalibrationRepository(conn)
    assert repo.count() == 0


def test_migration_refuses_to_drop_non_empty_old_schema_calibration_table(tmp_path):
    """Same discipline as merge_colab_db.py's count check -- refuse loudly
    rather than silently discarding real rated data, if this ever runs
    against a DB where rating actually happened under the old schema."""
    db_path = tmp_path / "old.db"
    raw_conn = _create_old_shape_calibration_table(db_path)
    raw_conn.execute(
        "INSERT INTO songs (fma_track_id, filepath, title, artist, genre_top, duration_sec) "
        "VALUES (1, '/x.mp3', 'Song', 'Artist', 'Rock', 30.0)"
    )
    raw_conn.execute(
        "INSERT INTO segments (song_id, start_sec, end_sec, segment_index) VALUES (1, 0.0, 5.0, 0), (1, 5.0, 10.0, 1)"
    )
    raw_conn.execute(
        "INSERT INTO calibration_ratings (segment_a_id, segment_b_id, rating) VALUES (1, 2, 4.0)"
    )
    raw_conn.commit()
    raw_conn.close()

    with pytest.raises(RuntimeError):
        init_db(db_path)
