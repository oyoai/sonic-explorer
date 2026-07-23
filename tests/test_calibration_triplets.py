import numpy as np
import pytest

from sonic_explorer.evaluation.calibration_triplets import generate_calibration_triplets
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


def add_song(song_repo, embedding_repo, track_id, vector, facet_name="sound"):
    song = Song(
        filepath=f"/data/audio/{track_id}.mp3", fma_track_id=track_id, title=f"Song {track_id}",
        artist="Artist", genre_top="Rock", duration_sec=10.0,
    )
    song_id = song_repo.add_song(song)
    [seg_id] = song_repo.add_segments(song_id, [Segment(song_id=song_id, start_sec=0.0, end_sec=5.0, segment_index=0)])
    embedding_repo.add_vector(facet_name, seg_id, vector)
    return song_id, seg_id


def test_generate_calibration_triplets_no_self_matches(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(0)
    for i in range(60):
        add_song(song_repo, embedding_repo, i, rng.normal(size=4).astype(np.float32))

    triplets = generate_calibration_triplets(song_repo, embedding_repo, facet_name="sound", n_triplets=30)

    for t in triplets:
        assert t.segment_x_id != t.segment_a_id
        assert t.segment_x_id != t.segment_b_id
        assert t.segment_a_id != t.segment_b_id


def test_generate_calibration_triplets_no_duplicate_triplets(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(1)
    for i in range(60):
        add_song(song_repo, embedding_repo, i, rng.normal(size=4).astype(np.float32))

    triplets = generate_calibration_triplets(song_repo, embedding_repo, facet_name="sound", n_triplets=30)

    seen = set()
    for t in triplets:
        key = (t.segment_x_id, min(t.segment_a_id, t.segment_b_id), max(t.segment_a_id, t.segment_b_id))
        assert key not in seen
        seen.add(key)


def test_generate_calibration_triplets_respects_requested_count(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(2)
    for i in range(60):
        add_song(song_repo, embedding_repo, i, rng.normal(size=4).astype(np.float32))

    triplets = generate_calibration_triplets(song_repo, embedding_repo, facet_name="sound", n_triplets=25)

    assert len(triplets) == 25


def test_generate_calibration_triplets_uses_all_three_band_combinations(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(3)
    for i in range(60):
        add_song(song_repo, embedding_repo, i, rng.normal(size=4).astype(np.float32))

    triplets = generate_calibration_triplets(song_repo, embedding_repo, facet_name="sound", n_triplets=30)

    band_combos = {frozenset((t.band_a, t.band_b)) for t in triplets}
    assert frozenset(("high", "medium")) in band_combos
    assert frozenset(("high", "random")) in band_combos
    assert frozenset(("medium", "random")) in band_combos


def test_generate_calibration_triplets_randomizes_ab_slot(repos):
    """The "closer" band of a pair must not always land in slot A -- a rater
    could otherwise learn to just always pick one side."""
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(4)
    for i in range(60):
        add_song(song_repo, embedding_repo, i, rng.normal(size=4).astype(np.float32))

    triplets = generate_calibration_triplets(song_repo, embedding_repo, facet_name="sound", n_triplets=30)

    high_medium = [t for t in triplets if frozenset((t.band_a, t.band_b)) == frozenset(("high", "medium"))]
    assert any(t.band_a == "high" for t in high_medium)
    assert any(t.band_a == "medium" for t in high_medium)


def test_generate_calibration_triplets_handles_no_embedded_segments(repos):
    song_repo, embedding_repo = repos
    triplets = generate_calibration_triplets(song_repo, embedding_repo, facet_name="sound")
    assert triplets == []


def test_generate_calibration_triplets_caps_gracefully_when_library_too_small(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(5)
    for i in range(3):
        add_song(song_repo, embedding_repo, i, rng.normal(size=4).astype(np.float32))

    # requesting far more triplets than a 3-segment library can supply must not hang or crash
    triplets = generate_calibration_triplets(song_repo, embedding_repo, facet_name="sound", n_triplets=50)
    assert len(triplets) <= 3
