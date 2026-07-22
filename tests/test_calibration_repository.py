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
def two_segments(conn):
    song_repo = SongRepository(conn)
    song = Song(filepath="/a.mp3", fma_track_id=1, title="A", artist="Artist", genre_top="Rock", duration_sec=30.0)
    song_id = song_repo.add_song(song)
    seg_ids = song_repo.add_segments(
        song_id,
        [
            Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0),
            Segment(song_id=song_id, start_sec=5.0, end_sec=10.0, segment_index=1),
        ],
    )
    return seg_ids


def test_add_rating_and_count(conn, two_segments):
    repo = CalibrationRepository(conn)
    seg_a, seg_b = two_segments

    repo.add_rating(seg_a, seg_b, rating=4.0, rater="offi")

    assert repo.count() == 1
    ratings = repo.get_all_ratings()
    assert ratings[0]["segment_a_id"] == seg_a
    assert ratings[0]["segment_b_id"] == seg_b
    assert ratings[0]["rating"] == 4.0
    assert ratings[0]["rater"] == "offi"


def test_rated_pair_ids_includes_both_orderings(conn, two_segments):
    repo = CalibrationRepository(conn)
    seg_a, seg_b = two_segments

    repo.add_rating(seg_a, seg_b, rating=2.0)

    pairs = repo.rated_pair_ids()
    assert (seg_a, seg_b) in pairs
    assert (seg_b, seg_a) in pairs


def test_count_with_no_ratings_is_zero(conn):
    repo = CalibrationRepository(conn)
    assert repo.count() == 0
    assert repo.get_all_ratings() == []
    assert repo.rated_pair_ids() == set()


def test_add_rating_without_rater_is_optional(conn, two_segments):
    repo = CalibrationRepository(conn)
    seg_a, seg_b = two_segments

    repo.add_rating(seg_a, seg_b, rating=3.5)

    assert repo.get_all_ratings()[0]["rater"] is None
