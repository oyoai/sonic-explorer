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

    name = "sound"
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


class FakeHarmonyFacet(FakeSoundFacet):
    """A second fake facet, distinct name, for exercising the multi-facet path."""

    name = "harmony"
    dim = 6


class CrashingFacet(FakeSoundFacet):
    """Raises on a specific call (1-indexed, one call per song) to simulate a
    process dying mid-run -- e.g. a Colab disconnect -- between checkpoints."""

    def __init__(self, crash_on_call: int):
        super().__init__()
        self.crash_on_call = crash_on_call

    def embed_batch(self, audio_windows, sr, batch_size=8):
        if self.call_count == self.crash_on_call - 1:
            raise RuntimeError("simulated crash")
        return super().embed_batch(audio_windows, sr, batch_size)


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

    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facets=[facet])

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

    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facets=[facet])
    first_index_size = embedding_repo.index_size("sound")
    assert facet.call_count > 0

    # simulate a resumed run: same manifest, same repos (as if DB/index were reloaded from Drive)
    facet.call_count = 0
    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facets=[facet])

    assert embedding_repo.index_size("sound") == first_index_size
    # every track was already fully embedded -- run_batch_embedding should skip the
    # audio load + embed_batch call entirely for both, per the get_song_by_fma_track_id check
    assert facet.call_count == 0


def test_get_vector_and_search_after_batch_embedding(repos, curated_audio):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio
    facet = FakeSoundFacet()

    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facets=[facet])

    song_a = song_repo.get_song_by_fma_track_id(1)
    query_seg = song_a.segments[0]
    query_vec = embedding_repo.get_vector("sound", query_seg.id)

    results = embedding_repo.search("sound", query_vec, k=3)
    assert results[0][0] == query_seg.id
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_crash_between_checkpoints_leaves_no_orphaned_done_status(repos, tmp_path):
    """Regression test for a real bug: mark_done() used to fire immediately per
    vector, decoupled from when the FAISS index was actually saved to disk. A
    crash between a checkpoint's save_index() and the next one left segment rows
    permanently marked 'done' with no corresponding vector in the persisted index
    -- silent data loss that only surfaced later as a FAISS reconstruct() crash."""
    song_repo, embedding_repo = repos
    audio_dir = tmp_path / "audio3"
    audio_dir.mkdir()
    make_sine_wav(audio_dir / "1.wav", duration_sec=12.0, freq=440.0)
    make_sine_wav(audio_dir / "2.wav", duration_sec=12.0, freq=550.0)
    make_sine_wav(audio_dir / "3.wav", duration_sec=12.0, freq=660.0)
    manifest = pd.DataFrame([
        {"track_id": 1, "genre_top": "Rock", "title": "Song A", "artist": "Artist A", "relative_path": "1.wav"},
        {"track_id": 2, "genre_top": "Jazz", "title": "Song B", "artist": "Artist B", "relative_path": "2.wav"},
        {"track_id": 3, "genre_top": "Folk", "title": "Song C", "artist": "Artist C", "relative_path": "3.wav"},
    ])
    facet = CrashingFacet(crash_on_call=3)  # crashes while processing song 3

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_batch_embedding(
            manifest, audio_dir, song_repo, embedding_repo, facets=[facet], checkpoint_every=2
        )

    # songs 1 & 2 were checkpointed (save_index + mark_done) before the crash
    song_a = song_repo.get_song_by_fma_track_id(1)
    song_b = song_repo.get_song_by_fma_track_id(2)
    for seg in song_a.segments + song_b.segments:
        assert embedding_repo.status(seg.id, "sound") == "done"

    # song 3's segment rows may exist (added before embed_batch was called), but
    # none should be marked 'done' -- the crash happened before add_to_index/mark_done
    song_c = song_repo.get_song_by_fma_track_id(3)
    if song_c is not None:
        for seg in song_c.segments:
            assert embedding_repo.status(seg.id, "sound") != "done"

    # the critical invariant: every segment marked 'done' actually has a vector in
    # the FAISS index -- no orphans, regardless of when the crash happened
    all_songs = [s for s in [song_a, song_b, song_c] if s is not None]
    for song in all_songs:
        for seg in song.segments:
            if embedding_repo.status(seg.id, "sound") == "done":
                embedding_repo.get_vector("sound", seg.id)  # must not raise


def test_multiple_facets_embedded_from_one_shared_audio_load(repos, curated_audio):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio
    sound = FakeSoundFacet()
    harmony = FakeHarmonyFacet()

    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facets=[sound, harmony])

    song_a = song_repo.get_song_by_fma_track_id(1)
    for seg in song_a.segments:
        assert embedding_repo.status(seg.id, "sound") == "done"
        assert embedding_repo.status(seg.id, "harmony") == "done"
    assert embedding_repo.index_size("sound") == embedding_repo.index_size("harmony")


def test_multiple_facets_independent_compute_once_tracking(repos, curated_audio):
    """If sound is already fully embedded but harmony isn't, a re-run must still
    reload audio (harmony needs it) while skipping re-embedding sound."""
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio
    sound = FakeSoundFacet()

    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facets=[sound])
    sound_index_size = embedding_repo.index_size("sound")

    sound.call_count = 0
    harmony = FakeHarmonyFacet()
    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facets=[sound, harmony])

    assert sound.call_count == 0  # sound was already done for every segment -- never re-embedded
    assert harmony.call_count > 0  # harmony had to actually run
    assert embedding_repo.index_size("sound") == sound_index_size  # unchanged
    assert embedding_repo.index_size("harmony") == sound_index_size  # same segments, now both facets present


def test_save_and_load_index_round_trips_through_artifacts_dir(repos, curated_audio):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio
    facet = FakeSoundFacet()

    run_batch_embedding(manifest, audio_dir, song_repo, embedding_repo, facets=[facet])
    embedding_repo.save_index("sound")

    index_path = embedding_repo.artifacts_dir / "sound.index"
    assert index_path.exists()

    fresh_repo = EmbeddingRepository(embedding_repo.conn, artifacts_dir=embedding_repo.artifacts_dir)
    fresh_repo.load_index("sound")
    assert fresh_repo.index_size("sound") == embedding_repo.index_size("sound")
