import numpy as np
import pytest

from sonic_explorer.evaluation.blend_weight_regression import compute_blend_weights
from sonic_explorer.facets.registry import default_registry
from sonic_explorer.models import Segment, Song
from sonic_explorer.repository.calibration_repository import CalibrationRepository
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

ALL_FACETS = default_registry().names()


@pytest.fixture
def conn():
    connection = init_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def repos(conn):
    return SongRepository(conn), EmbeddingRepository(conn), CalibrationRepository(conn)


def _add_segment(song_repo, track_id) -> int:
    song = Song(
        filepath=f"/data/{track_id}.mp3", fma_track_id=track_id, title=f"Song {track_id}",
        artist="Artist", genre_top="Rock", duration_sec=10.0,
    )
    song_id = song_repo.add_song(song)
    [seg_id] = song_repo.add_segments(song_id, [Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0)])
    return seg_id


def _embed_all_facets(embedding_repo, seg_id, vector):
    for facet_name in ALL_FACETS:
        embedding_repo.add_vector(facet_name, seg_id, vector.astype(np.float32))


def test_compute_blend_weights_reports_sample_size_and_rater_count(repos):
    song_repo, embedding_repo, calibration_repo = repos
    x, a, b = _add_segment(song_repo, 1), _add_segment(song_repo, 2), _add_segment(song_repo, 3)
    calibration_repo.add_choice(x, a, b, choice="a", rater="profile1")
    calibration_repo.add_choice(x, a, b, choice="b", rater="profile2")

    result = compute_blend_weights(calibration_repo, song_repo, embedding_repo)

    assert result.n_ratings == 2
    assert result.n_raters == 2


def test_compute_blend_weights_handles_zero_ratings(repos):
    song_repo, embedding_repo, calibration_repo = repos

    result = compute_blend_weights(calibration_repo, song_repo, embedding_repo)

    assert result.n_ratings == 0
    assert result.n_raters == 0
    assert result.regression_weights is None
    assert result.regression_note is not None
    assert all(np.isnan(v) for v in result.agreement_rate.values())


def test_compute_blend_weights_agreement_rate_is_perfect_when_facet_always_matches_choice(repos):
    """Reference and A share the same vector (similarity 1.0); B is
    orthogonal (similarity 0.0) on "sound" specifically -- sound's diff is
    always positive, so a rater always picking "a" should show sound at
    100% agreement."""
    song_repo, embedding_repo, calibration_repo = repos
    x, a, b = _add_segment(song_repo, 1), _add_segment(song_repo, 2), _add_segment(song_repo, 3)
    shared = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    orthogonal = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    embedding_repo.add_vector("sound", x, shared)
    embedding_repo.add_vector("sound", a, shared)
    embedding_repo.add_vector("sound", b, orthogonal)

    calibration_repo.add_choice(x, a, b, choice="a", rater="profile1")

    result = compute_blend_weights(calibration_repo, song_repo, embedding_repo)

    assert result.agreement_rate["sound"] == pytest.approx(1.0)
    assert result.agreement_n["sound"] == 1


def test_compute_blend_weights_agreement_rate_ties_get_half_credit(repos):
    song_repo, embedding_repo, calibration_repo = repos
    x, a, b = _add_segment(song_repo, 1), _add_segment(song_repo, 2), _add_segment(song_repo, 3)
    vec = np.array([1.0, 0.0], dtype=np.float32)
    embedding_repo.add_vector("sound", x, vec)
    embedding_repo.add_vector("sound", a, vec)
    embedding_repo.add_vector("sound", b, vec)  # identical -- a genuine tie, diff == 0

    calibration_repo.add_choice(x, a, b, choice="a", rater="profile1")

    result = compute_blend_weights(calibration_repo, song_repo, embedding_repo)

    assert result.agreement_rate["sound"] == pytest.approx(0.5)


def test_compute_blend_weights_facet_missing_an_embedding_is_excluded_not_counted_as_disagreement(repos):
    song_repo, embedding_repo, calibration_repo = repos
    x, a, b = _add_segment(song_repo, 1), _add_segment(song_repo, 2), _add_segment(song_repo, 3)
    # No "harmony" vectors added for any of x/a/b at all.
    calibration_repo.add_choice(x, a, b, choice="a", rater="profile1")

    result = compute_blend_weights(calibration_repo, song_repo, embedding_repo)

    assert result.agreement_n["harmony"] == 0
    assert np.isnan(result.agreement_rate["harmony"])


def test_compute_blend_weights_regression_not_fit_with_too_few_complete_rows(repos):
    song_repo, embedding_repo, calibration_repo = repos
    x, a, b = _add_segment(song_repo, 1), _add_segment(song_repo, 2), _add_segment(song_repo, 3)
    _embed_all_facets(embedding_repo, x, np.array([1.0, 0.0], dtype=np.float32))
    _embed_all_facets(embedding_repo, a, np.array([1.0, 0.0], dtype=np.float32))
    _embed_all_facets(embedding_repo, b, np.array([0.0, 1.0], dtype=np.float32))
    calibration_repo.add_choice(x, a, b, choice="a", rater="profile1")

    result = compute_blend_weights(calibration_repo, song_repo, embedding_repo)

    assert result.regression_weights is None
    assert "too few" in result.regression_note.lower()


def test_compute_blend_weights_regression_not_fit_when_all_choices_are_the_same(repos):
    song_repo, embedding_repo, calibration_repo = repos
    rng = np.random.default_rng(0)
    for i in range(12):
        x, a, b = _add_segment(song_repo, i * 3 + 1), _add_segment(song_repo, i * 3 + 2), _add_segment(song_repo, i * 3 + 3)
        _embed_all_facets(embedding_repo, x, rng.normal(size=4).astype(np.float32))
        _embed_all_facets(embedding_repo, a, rng.normal(size=4).astype(np.float32))
        _embed_all_facets(embedding_repo, b, rng.normal(size=4).astype(np.float32))
        calibration_repo.add_choice(x, a, b, choice="a", rater="profile1")  # always "a"

    result = compute_blend_weights(calibration_repo, song_repo, embedding_repo)

    assert result.regression_weights is None
    assert "same side" in result.regression_note.lower()


def test_compute_blend_weights_regression_fits_with_enough_varied_data(repos):
    song_repo, embedding_repo, calibration_repo = repos
    rng = np.random.default_rng(1)
    for i in range(12):
        x, a, b = _add_segment(song_repo, i * 3 + 1), _add_segment(song_repo, i * 3 + 2), _add_segment(song_repo, i * 3 + 3)
        _embed_all_facets(embedding_repo, x, rng.normal(size=4).astype(np.float32))
        _embed_all_facets(embedding_repo, a, rng.normal(size=4).astype(np.float32))
        _embed_all_facets(embedding_repo, b, rng.normal(size=4).astype(np.float32))
        calibration_repo.add_choice(x, a, b, choice="a" if i % 2 == 0 else "b", rater="profile1")

    result = compute_blend_weights(calibration_repo, song_repo, embedding_repo)

    assert result.regression_weights is not None
    assert set(result.regression_weights.keys()) == set(ALL_FACETS)
    assert all(isinstance(v, float) for v in result.regression_weights.values())
    assert result.regression_note is None
