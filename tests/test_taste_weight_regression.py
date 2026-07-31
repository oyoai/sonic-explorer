import pytest

from sonic_explorer.analysis.song_dna import AXES, fit_normalizer
from sonic_explorer.evaluation.taste_weight_regression import compute_taste_comparison, compute_taste_weights
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
def repos(conn):
    return SongRepository(conn), TasteRepository(conn)


def _add_song_with_dna(song_repo, track_id, tempo_bpm, energy, brightness, harmonic_complexity=0.5, rhythmic_density=0.5) -> tuple[int, int]:
    """Real fabricated DNA -- test fixture data, standard practice (unlike
    seeding the live app's own DB with fake ratings, which this feature
    deliberately never does -- see the module's own docstring)."""
    song = Song(
        filepath=f"/data/{track_id}.mp3", fma_track_id=track_id, title=f"Song {track_id}",
        artist="Artist", genre_top="Rock", duration_sec=10.0,
    )
    song_id = song_repo.add_song(song)
    song_repo.update_song_dna(song_id, tempo_bpm, energy, brightness, harmonic_complexity, rhythmic_density)
    [seg_id] = song_repo.add_segments(song_id, [Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0)])
    return song_id, seg_id


def _normalizer_for(song_repo):
    return fit_normalizer([{axis: getattr(s, axis) for axis in AXES} for s in song_repo.list_songs()])


def test_compute_taste_weights_reports_sample_size_rater_count_and_like_split(repos):
    song_repo, taste_repo = repos
    _, seg1 = _add_song_with_dna(song_repo, 1, 120.0, 0.2, 2000.0)
    _, seg2 = _add_song_with_dna(song_repo, 2, 90.0, 0.05, 1000.0)
    taste_repo.add_rating(seg1, liked=True, rater="profile1")
    taste_repo.add_rating(seg2, liked=False, rater="profile2")

    result = compute_taste_weights(taste_repo, song_repo, _normalizer_for(song_repo))

    assert result.n_ratings == 2
    assert result.n_raters == 2
    assert result.n_liked == 1
    assert result.n_disliked == 1


def test_compute_taste_weights_handles_zero_ratings(repos):
    song_repo, taste_repo = repos

    result = compute_taste_weights(taste_repo, song_repo, _normalizer_for(song_repo))

    assert result.n_ratings == 0
    assert result.n_raters == 0
    assert result.regression_weights is None
    assert result.regression_note is not None
    assert "too few" in result.regression_note.lower()


def test_compute_taste_weights_regression_not_fit_with_too_few_ratings(repos):
    song_repo, taste_repo = repos
    _, seg1 = _add_song_with_dna(song_repo, 1, 120.0, 0.2, 2000.0)
    _, seg2 = _add_song_with_dna(song_repo, 2, 90.0, 0.05, 1000.0)
    taste_repo.add_rating(seg1, liked=True, rater="profile1")
    taste_repo.add_rating(seg2, liked=False, rater="profile1")

    result = compute_taste_weights(taste_repo, song_repo, _normalizer_for(song_repo))

    assert result.regression_weights is None
    assert "too few" in result.regression_note.lower()


def test_compute_taste_weights_regression_not_fit_when_every_rating_is_the_same(repos):
    song_repo, taste_repo = repos
    for i in range(12):
        _, seg = _add_song_with_dna(song_repo, i + 1, 100.0 + i, 0.1 + i * 0.01, 1500.0 + i * 10)
        taste_repo.add_rating(seg, liked=True, rater="profile1")  # always liked

    result = compute_taste_weights(taste_repo, song_repo, _normalizer_for(song_repo))

    assert result.regression_weights is None
    assert "same" in result.regression_note.lower()


def test_compute_taste_weights_regression_fits_with_enough_varied_data(repos):
    song_repo, taste_repo = repos
    for i in range(12):
        _, seg = _add_song_with_dna(song_repo, i + 1, 60.0 + i * 10, 0.02 + i * 0.03, 500.0 + i * 200)
        taste_repo.add_rating(seg, liked=(i % 2 == 0), rater="profile1")

    result = compute_taste_weights(taste_repo, song_repo, _normalizer_for(song_repo))

    assert result.regression_weights is not None
    assert set(result.regression_weights.keys()) == set(AXES)
    assert all(isinstance(v, float) for v in result.regression_weights.values())
    assert result.regression_note is None


def test_compute_taste_comparison_reports_nothing_to_compare_with_zero_or_one_raters(repos):
    song_repo, taste_repo = repos

    result = compute_taste_comparison(taste_repo, song_repo, _normalizer_for(song_repo))
    assert result.per_rater == []
    assert result.note is not None
    assert "0 rater" in result.note

    _, seg1 = _add_song_with_dna(song_repo, 1, 120.0, 0.2, 2000.0)
    taste_repo.add_rating(seg1, liked=True, rater="profile1")

    result = compute_taste_comparison(taste_repo, song_repo, _normalizer_for(song_repo))
    assert len(result.per_rater) == 1
    assert result.note is not None
    assert "1 rater" in result.note


def test_compute_taste_comparison_reports_a_real_per_rater_breakdown_with_two_raters(repos):
    song_repo, taste_repo = repos
    _, seg1 = _add_song_with_dna(song_repo, 1, 120.0, 0.2, 2000.0)
    _, seg2 = _add_song_with_dna(song_repo, 2, 90.0, 0.05, 1000.0)
    _, seg3 = _add_song_with_dna(song_repo, 3, 150.0, 0.3, 3000.0)
    taste_repo.add_rating(seg1, liked=True, rater="profile1")
    taste_repo.add_rating(seg2, liked=False, rater="profile1")
    taste_repo.add_rating(seg3, liked=True, rater="profile2")

    result = compute_taste_comparison(taste_repo, song_repo, _normalizer_for(song_repo))

    assert result.note is None
    assert [s.rater for s in result.per_rater] == ["profile1", "profile2"]  # sorted, stable order

    profile1_summary = result.per_rater[0]
    assert profile1_summary.n_ratings == 2
    assert profile1_summary.n_liked == 1
    assert profile1_summary.n_disliked == 1
    assert profile1_summary.liked_pct == pytest.approx(0.5)
    assert profile1_summary.mean_liked_dna is not None
    assert set(profile1_summary.mean_liked_dna.keys()) == set(AXES)

    profile2_summary = result.per_rater[1]
    assert profile2_summary.n_ratings == 1
    assert profile2_summary.liked_pct == pytest.approx(1.0)


def test_compute_taste_comparison_mean_liked_dna_is_none_when_a_rater_liked_nothing(repos):
    song_repo, taste_repo = repos
    _, seg1 = _add_song_with_dna(song_repo, 1, 120.0, 0.2, 2000.0)
    _, seg2 = _add_song_with_dna(song_repo, 2, 90.0, 0.05, 1000.0)
    taste_repo.add_rating(seg1, liked=False, rater="profile1")
    taste_repo.add_rating(seg2, liked=True, rater="profile2")

    result = compute_taste_comparison(taste_repo, song_repo, _normalizer_for(song_repo))

    profile1_summary = next(s for s in result.per_rater if s.rater == "profile1")
    assert profile1_summary.mean_liked_dna is None
