"""Integration test for pipeline/embed_tags.py -- synthetic audio + a fake
sound-tags facet, so this exercises segmentation, repository, and
resumability/checkpoint logic without needing torch/transformers/AST/CLAP."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sonic_explorer.config import CLAP_SR
from sonic_explorer.facets.tags import NoTagsDetected
from sonic_explorer.models import Song
from sonic_explorer.pipeline.embed_tags import run_batch_tags_embedding
from sonic_explorer.pipeline.segment import segment_song
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository


class FakeTagsFacet:
    """Duck-typed like SoundTagsFacet: embed(audio, sr) -> vector, or raises
    NoTagsDetected for windows this fake treats as tag-less (identified by a
    marker frequency baked into the synthetic audio, since a real fake AST
    tagger isn't worth building)."""

    name = "sound_tags"
    dim = 8

    def __init__(self, no_tags_freq=None):
        self.no_tags_freq = no_tags_freq
        self.call_count = 0
        self._counter = 0

    def embed(self, audio, sr):
        self.call_count += 1
        if self.no_tags_freq is not None and _dominant_freq_marker(audio, sr) == self.no_tags_freq:
            raise NoTagsDetected("fake: no tags for this marker frequency")
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[self._counter % self.dim] = 1.0
        self._counter += 1
        return vec


def _dominant_freq_marker(audio, sr):
    """Cheap stand-in for 'what frequency was this window generated at' --
    tests pass distinguishable sine frequencies and check via FFT peak. Must
    use the actual sr embed() receives (AST_SAMPLE_RATE, 16kHz) -- the
    pipeline resamples from the wav file's real sample rate (CLAP_SR, 48kHz)
    before calling embed(), so hardcoding CLAP_SR here would compute the
    wrong frequency axis entirely."""
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    return round(freqs[np.argmax(spectrum)])


def make_sine_wav(path, duration_sec, freq=440.0, sr=CLAP_SR):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, sr)
    return audio


@pytest.fixture
def repos(tmp_path):
    conn = init_db(tmp_path / "artifacts" / "sonic_explorer.db")
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=tmp_path / "artifacts")
    return song_repo, embedding_repo


@pytest.fixture
def curated_audio_with_songs(tmp_path, repos):
    """Pre-populates songs+segments in the DB (as if the sound-facet batch
    job already ran) -- run_batch_tags_embedding never creates songs itself,
    only ever adds facet vectors for songs that already exist."""
    song_repo, _ = repos
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    make_sine_wav(audio_dir / "1.wav", duration_sec=12.0, freq=440.0)
    make_sine_wav(audio_dir / "2.wav", duration_sec=8.0, freq=880.0)

    for track_id, duration in [(1, 12.0), (2, 8.0)]:
        song = Song(filepath="x", fma_track_id=track_id, title=f"Song {track_id}", artist="A",
                    genre_top="Rock", duration_sec=duration)
        song_id = song_repo.add_song(song)
        segments = segment_song(song_id, duration)
        song_repo.add_segments(song_id, segments)

    manifest = pd.DataFrame([
        {"track_id": 1, "relative_path": "1.wav"},
        {"track_id": 2, "relative_path": "2.wav"},
    ])
    return audio_dir, manifest


def test_run_batch_tags_embedding_embeds_all_segments(repos, curated_audio_with_songs):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    facet = FakeTagsFacet()

    run_batch_tags_embedding(manifest, audio_dir, song_repo, embedding_repo, facet)

    song_a = song_repo.get_song_by_fma_track_id(1)
    for seg in song_a.segments:
        assert embedding_repo.status(seg.id, "sound_tags") == "done"
        embedding_repo.get_vector("sound_tags", seg.id)  # must not raise -- actually persisted


def test_run_batch_tags_embedding_skips_songs_not_in_db(repos, curated_audio_with_songs):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    manifest = pd.concat([manifest, pd.DataFrame([{"track_id": 999, "relative_path": "missing.wav"}])])

    failed = run_batch_tags_embedding(manifest, audio_dir, song_repo, embedding_repo, FakeTagsFacet())
    assert failed == []  # skipped, not failed -- track 999 was never in the DB


def test_run_batch_tags_embedding_is_idempotent(repos, curated_audio_with_songs):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    facet = FakeTagsFacet()

    run_batch_tags_embedding(manifest, audio_dir, song_repo, embedding_repo, facet)
    first_size = embedding_repo.index_size("sound_tags")
    first_calls = facet.call_count

    run_batch_tags_embedding(manifest, audio_dir, song_repo, embedding_repo, facet)
    assert facet.call_count == first_calls  # already fully embedded -- embed() never re-runs
    assert embedding_repo.index_size("sound_tags") == first_size


def test_run_batch_tags_embedding_marks_no_tags_segments_as_skipped_not_embedded(repos, curated_audio_with_songs):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    facet = FakeTagsFacet(no_tags_freq=440)  # song 1's whole clip is 440Hz

    run_batch_tags_embedding(manifest, audio_dir, song_repo, embedding_repo, facet)

    song_a = song_repo.get_song_by_fma_track_id(1)
    for seg in song_a.segments:
        assert embedding_repo.status(seg.id, "sound_tags") == "skipped"
    song_b = song_repo.get_song_by_fma_track_id(2)
    for seg in song_b.segments:
        assert embedding_repo.status(seg.id, "sound_tags") == "done"


def test_run_batch_tags_embedding_skipped_counts_as_finished_for_resumability(repos, curated_audio_with_songs):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    facet = FakeTagsFacet(no_tags_freq=440)

    run_batch_tags_embedding(manifest, audio_dir, song_repo, embedding_repo, facet)
    calls_after_first_run = facet.call_count

    run_batch_tags_embedding(manifest, audio_dir, song_repo, embedding_repo, facet)
    assert facet.call_count == calls_after_first_run  # both songs already fully resolved -- no re-tagging


def test_run_batch_tags_embedding_isolates_per_song_failures(repos, curated_audio_with_songs):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs

    class FlakyFacet(FakeTagsFacet):
        def embed(self, audio, sr):
            if _dominant_freq_marker(audio, sr) == 880:  # song 2's marker frequency
                raise RuntimeError("simulated tagging failure")
            return super().embed(audio, sr)

    errors = []
    failed = run_batch_tags_embedding(
        manifest, audio_dir, song_repo, embedding_repo, FlakyFacet(),
        on_error=lambda track_id, exc: errors.append(track_id),
    )

    assert failed == [2]
    assert errors == [2]
    song_a = song_repo.get_song_by_fma_track_id(1)
    for seg in song_a.segments:
        assert embedding_repo.status(seg.id, "sound_tags") == "done"


def test_run_batch_tags_embedding_checkpoints_and_leaves_no_orphaned_done_status(repos, tmp_path):
    """A mid-batch failure on one song must not affect the durability of
    already-checkpointed songs before it."""
    song_repo, embedding_repo = repos
    audio_dir = tmp_path / "audio3"
    audio_dir.mkdir()
    make_sine_wav(audio_dir / "1.wav", duration_sec=12.0, freq=440.0)
    make_sine_wav(audio_dir / "2.wav", duration_sec=12.0, freq=550.0)
    make_sine_wav(audio_dir / "3.wav", duration_sec=12.0, freq=660.0)

    for track_id in (1, 2, 3):
        song = Song(filepath="x", fma_track_id=track_id, title=f"Song {track_id}", artist="A",
                    genre_top="Rock", duration_sec=12.0)
        song_id = song_repo.add_song(song)
        song_repo.add_segments(song_id, segment_song(song_id, 12.0))

    manifest = pd.DataFrame([
        {"track_id": 1, "relative_path": "1.wav"},
        {"track_id": 2, "relative_path": "2.wav"},
        {"track_id": 3, "relative_path": "3.wav"},
    ])

    class FailThirdSongFacet(FakeTagsFacet):
        def embed(self, audio, sr):
            if _dominant_freq_marker(audio, sr) == 660:  # song 3's marker frequency
                raise RuntimeError("simulated crash")
            return super().embed(audio, sr)

    errors = []
    failed = run_batch_tags_embedding(
        manifest, audio_dir, song_repo, embedding_repo, FailThirdSongFacet(),
        checkpoint_every=2, on_error=lambda track_id, exc: errors.append(track_id),
    )

    assert failed == [3]
    assert errors == [3]

    for track_id in (1, 2):
        song = song_repo.get_song_by_fma_track_id(track_id)
        for seg in song.segments:
            assert embedding_repo.status(seg.id, "sound_tags") == "done"
            embedding_repo.get_vector("sound_tags", seg.id)  # must not raise -- vector actually persisted

    song3 = song_repo.get_song_by_fma_track_id(3)
    for seg in song3.segments:
        assert embedding_repo.status(seg.id, "sound_tags") != "done"
