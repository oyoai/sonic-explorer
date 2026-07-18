"""Integration test for the actual batch pipeline notebooks/02 runs -- synthetic
audio + a fake (non-CLAP) facet, so this exercises real segmentation, repository,
and resumability logic without needing torch/CLAP/GPU."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sonic_explorer.config import CLAP_SR
from sonic_explorer.pipeline.embed_library import run_batch_embedding
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository


class FakeSoundFacet:
    """Duck-types SoundFacet.embed_batch without loading any real model. Assigns each
    window a distinct one-hot direction via a running counter -- a stationary test
    tone's raw content (mean/std) is near-identical across windows since the hop is
    close to an integer number of periods, so content-derived fake vectors would
    collapse to duplicates and defeat any retrieval-identity assertions."""

    dim = 8

    def __init__(self):
        self.call_count = 0
        self._counter = 0

    def embed_batch(self, audio_windows, sr, batch_size=8):
        self.call_count += 1
        vectors = []
        for _ in audio_windows:
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[self._counter % self.dim] = 1.0
            vectors.append(vec)
            self._counter += 1
        return np.stack(vectors)


def make_sine_wav(path, duration_sec, freq=440.0, sr=CLAP_SR):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), audio, sr)


@pytest.fixture
def repos(tmp_path):
    conn = init_db(tmp_path / "artifacts" / "sonic_explorer.db")
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=tmp_path / "artifacts")
    return song_repo, embedding_repo


@pytest.fixture
def curated_audio(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    make_sine_wav(audio_dir / "1.wav", duration_sec=12.0, freq=440.0)
    make_sine_wav(audio_dir / "2.wav", duration_sec=8.0, freq=880.0)
    manifest = pd.DataFrame([
        {"track_id": 1, "genre_top": "Rock", "title": "Song A", "artist": "Artist A", "relative_path": "1.wav"},
        {"track_id": 2, "genre_top": "Jazz", "title": "Song B", "artist": "Artist B", "relative_path": "2.wav"},
    ])
    return audio_dir, manifest


def test_batch_embedding_populates_db_and_index(repos, curated_audio):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio
    facet = FakeSoundFacet()

    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facet, facet_name="sound")

    songs = song_repo.list_songs()
    assert len(songs) == 2

    # 12s @ 5s window / 2.5s hop -> 3 segments; 8s -> 1 segment (window=5<=8: starts 0,2.5,5 -> 3? check below)
    song_a = song_repo.get_song_by_fma_track_id(1)
    song_b = song_repo.get_song_by_fma_track_id(2)
    assert len(song_a.segments) > 0
    assert len(song_b.segments) > 0

    total_segments = len(song_a.segments) + len(song_b.segments)
    assert embedding_repo.index_size("sound") == total_segments

    for seg in song_a.segments:
        assert embedding_repo.status(seg.id, "sound") == "done"


def test_batch_embedding_is_idempotent_and_skips_audio_reload_on_resume(repos, curated_audio):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio
    facet = FakeSoundFacet()

    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facet, facet_name="sound")
    first_index_size = embedding_repo.index_size("sound")
    assert facet.call_count > 0

    # simulate a resumed run: same manifest, same repos (as if DB/index were reloaded from Drive)
    facet.call_count = 0
    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facet, facet_name="sound")

    assert embedding_repo.index_size("sound") == first_index_size
    # every track was already fully embedded -- run_batch_embedding should skip the
    # audio load + embed_batch call entirely for both, per the get_song_by_fma_track_id check
    assert facet.call_count == 0


def test_get_vector_and_search_after_batch_embedding(repos, curated_audio):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio
    facet = FakeSoundFacet()

    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facet, facet_name="sound")

    song_a = song_repo.get_song_by_fma_track_id(1)
    query_seg = song_a.segments[0]
    query_vec = embedding_repo.get_vector("sound", query_seg.id)

    results = embedding_repo.search("sound", query_vec, k=3)
    assert results[0][0] == query_seg.id
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_save_and_load_index_round_trips_through_artifacts_dir(repos, curated_audio):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio
    facet = FakeSoundFacet()

    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facet, facet_name="sound")
    embedding_repo.save_index("sound")

    index_path = embedding_repo.artifacts_dir / "sound.index"
    assert index_path.exists()

    fresh_repo = EmbeddingRepository(embedding_repo.conn, artifacts_dir=embedding_repo.artifacts_dir)
    fresh_repo.load_index("sound")
    assert fresh_repo.index_size("sound") == embedding_repo.index_size("sound")
