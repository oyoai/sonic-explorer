import pytest

from sonic_explorer.models import Segment, Song
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.song_repository import SongRepository
from sonic_explorer.repository.taste_repository import TasteRepository


@pytest.fixture
def conn():
    connection = init_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def one_segment(conn):
    song_repo = SongRepository(conn)
    song = Song(filepath="/a.mp3", fma_track_id=1, title="A", artist="Artist", genre_top="Rock", duration_sec=30.0)
    song_id = song_repo.add_song(song)
    seg_ids = song_repo.add_segments(
        song_id, [Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0)]
    )
    return seg_ids[0]


def test_add_rating_and_count(conn, one_segment):
    repo = TasteRepository(conn)

    repo.add_rating(one_segment, liked=True, rater="profile1")

    assert repo.count() == 1
    ratings = repo.get_all_ratings()
    assert ratings[0]["segment_id"] == one_segment
    assert ratings[0]["liked"] == 1
    assert ratings[0]["rater"] == "profile1"


def test_add_rating_disliked_stores_zero(conn, one_segment):
    repo = TasteRepository(conn)
    repo.add_rating(one_segment, liked=False, rater="profile1")
    assert repo.get_all_ratings()[0]["liked"] == 0


def test_add_rating_without_rater_is_optional(conn, one_segment):
    repo = TasteRepository(conn)
    repo.add_rating(one_segment, liked=True)
    assert repo.get_all_ratings()[0]["rater"] is None


def test_count_with_no_ratings_is_zero(conn):
    repo = TasteRepository(conn)
    assert repo.count() == 0
    assert repo.get_all_ratings() == []
    assert repo.rated_segment_ids() == set()


def test_rejects_unknown_table_name(conn):
    with pytest.raises(ValueError):
        TasteRepository(conn, table="taste_ratings; DROP TABLE songs")


def test_guest_table_is_fully_isolated_from_the_real_table(conn, one_segment):
    real_repo = TasteRepository(conn, table="taste_ratings")
    guest_repo = TasteRepository(conn, table="taste_ratings_guest")

    guest_repo.add_rating(one_segment, liked=True, rater="Guest / Test")

    assert real_repo.count() == 0
    assert guest_repo.count() == 1


def test_rated_segment_ids_filters_by_rater(conn, one_segment):
    """Essential for multiple real profiles: one profile rating a segment
    must not make it disappear from another profile's remaining queue."""
    repo = TasteRepository(conn)
    repo.add_rating(one_segment, liked=True, rater="profile1")

    assert one_segment in repo.rated_segment_ids(rater="profile1")
    assert one_segment not in repo.rated_segment_ids(rater="profile2")
    assert one_segment in repo.rated_segment_ids()  # no filter -- sees every rater's history
