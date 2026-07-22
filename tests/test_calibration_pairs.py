import numpy as np
import pytest

from sonic_explorer.evaluation.calibration_pairs import generate_calibration_pairs
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


def test_generate_calibration_pairs_respects_band_counts(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(0)
    for i in range(60):
        add_song(song_repo, embedding_repo, i, rng.normal(size=4).astype(np.float32))

    pairs = generate_calibration_pairs(song_repo, embedding_repo, facet_name="sound", n_high=10, n_medium=10, n_random=10)

    bands = [p.band for p in pairs]
    assert bands.count("high") == 10
    assert bands.count("medium") == 10
    assert bands.count("random") == 10


def test_generate_calibration_pairs_never_pairs_a_segment_with_itself(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(1)
    for i in range(30):
        add_song(song_repo, embedding_repo, i, rng.normal(size=4).astype(np.float32))

    pairs = generate_calibration_pairs(song_repo, embedding_repo, facet_name="sound", n_high=5, n_medium=5, n_random=5)

    for p in pairs:
        assert p.segment_a_id != p.segment_b_id


def test_generate_calibration_pairs_no_duplicate_pairs(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(2)
    for i in range(40):
        add_song(song_repo, embedding_repo, i, rng.normal(size=4).astype(np.float32))

    pairs = generate_calibration_pairs(song_repo, embedding_repo, facet_name="sound", n_high=15, n_medium=15, n_random=15)

    seen = set()
    for p in pairs:
        key = (min(p.segment_a_id, p.segment_b_id), max(p.segment_a_id, p.segment_b_id))
        assert key not in seen
        seen.add(key)


def test_generate_calibration_pairs_handles_no_embedded_segments(repos):
    song_repo, embedding_repo = repos
    pairs = generate_calibration_pairs(song_repo, embedding_repo, facet_name="sound")
    assert pairs == []


def test_generate_calibration_pairs_caps_gracefully_when_library_too_small(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(3)
    for i in range(3):
        add_song(song_repo, embedding_repo, i, rng.normal(size=4).astype(np.float32))

    # requesting far more pairs than a 3-segment library can supply must not hang or crash
    pairs = generate_calibration_pairs(song_repo, embedding_repo, facet_name="sound", n_high=50, n_medium=50, n_random=50)
    assert len(pairs) <= 3  # at most 3 unique cross-song pairs possible among 3 segments
