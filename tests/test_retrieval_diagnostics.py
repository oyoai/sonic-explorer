import numpy as np
import pytest

from sonic_explorer.evaluation.retrieval_diagnostics import song_level_score_distribution, top1_score_distribution
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
    return song_id


def test_top1_score_distribution_separates_tight_clusters_from_random(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(0)

    # two tight, well-separated clusters -- top-1 within a cluster should score
    # far higher than a random cross-cluster pair
    for i in range(6):
        add_song(song_repo, embedding_repo, i, rng.normal(loc=[10, 10], scale=0.05, size=2).astype(np.float32))
    for i in range(6, 12):
        add_song(song_repo, embedding_repo, i, rng.normal(loc=[-10, -10], scale=0.05, size=2).astype(np.float32))

    result = top1_score_distribution(song_repo, embedding_repo, facet_name="sound")

    assert result.n_queries == 12
    assert len(result.top1_scores) == 12
    assert len(result.random_pair_scores) == 12
    assert np.mean(result.top1_scores) > np.mean(result.random_pair_scores)


def test_top1_score_distribution_handles_no_embedded_segments(repos):
    song_repo, embedding_repo = repos
    result = top1_score_distribution(song_repo, embedding_repo, facet_name="sound")
    assert result.n_queries == 0
    assert result.top1_scores == []
    assert result.random_pair_scores == []


def test_top1_score_distribution_respects_sample_size(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(1)
    for i in range(20):
        add_song(song_repo, embedding_repo, i, rng.normal(size=2).astype(np.float32))

    result = top1_score_distribution(song_repo, embedding_repo, facet_name="sound", sample_size=5)
    assert result.n_queries == 5


def test_top1_score_distribution_single_song_has_no_neighbors(repos):
    song_repo, embedding_repo = repos
    add_song(song_repo, embedding_repo, 1, np.array([1.0, 0.0], dtype=np.float32))

    result = top1_score_distribution(song_repo, embedding_repo, facet_name="sound")

    assert result.n_queries == 1
    assert result.top1_scores == []
    assert result.random_pair_scores == []


def test_top1_score_distribution_reports_top1_top2_margin(repos):
    song_repo, embedding_repo = repos
    # three songs at increasing distance from the query direction -- margin
    # between the best and 2nd-best real neighbor should be measurable and positive
    add_song(song_repo, embedding_repo, 1, np.array([1.0, 0.0], dtype=np.float32))
    add_song(song_repo, embedding_repo, 2, np.array([0.9, 0.1], dtype=np.float32))
    add_song(song_repo, embedding_repo, 3, np.array([0.0, 1.0], dtype=np.float32))

    result = top1_score_distribution(song_repo, embedding_repo, facet_name="sound")

    assert len(result.top1_top2_margins) == 3
    assert all(m >= 0 for m in result.top1_top2_margins)


def test_song_level_score_distribution_separates_tight_clusters_from_random(repos):
    song_repo, embedding_repo = repos
    rng = np.random.default_rng(0)
    for i in range(6):
        add_song(song_repo, embedding_repo, i, rng.normal(loc=[10, 10], scale=0.05, size=2).astype(np.float32))
    for i in range(6, 12):
        add_song(song_repo, embedding_repo, i, rng.normal(loc=[-10, -10], scale=0.05, size=2).astype(np.float32))

    result = song_level_score_distribution(song_repo, embedding_repo, facet_name="sound")

    assert result.n_queries == 12
    assert len(result.top1_scores) == 12
    assert np.mean(result.top1_scores) > np.mean(result.random_pair_scores)


def test_song_level_score_distribution_handles_no_embedded_segments(repos):
    song_repo, embedding_repo = repos
    result = song_level_score_distribution(song_repo, embedding_repo, facet_name="sound")
    assert result.n_queries == 0
    assert result.top1_scores == []
