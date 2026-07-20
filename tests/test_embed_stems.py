"""Integration test for pipeline/embed_stems.py -- synthetic audio + fake
stem facets + a fake separator, so this exercises segmentation, repository,
and resumability/checkpoint logic without needing torch/demucs/GPU."""

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sonic_explorer.config import CLAP_SR
from sonic_explorer.models import Song
from sonic_explorer.pipeline.embed_stems import _has_meaningful_energy, run_batch_stem_embedding
from sonic_explorer.pipeline.segment import segment_song
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository


class FakeStemFacet:
    def __init__(self, name, dim=8):
        self.name = name
        self.dim = dim
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


def fake_separate_stems(audio, sr):
    """Returns 4 stems, each just a scaled copy of the input -- enough to
    exercise segmentation/embedding without real separation."""
    return {
        "vocal": audio * 0.9,
        "drums": audio * 0.8,
        "bass": audio * 0.7,
        "instrumental": audio * 0.6,
    }


def fake_separate_stems_silent_vocal(audio, sr):
    """Same as fake_separate_stems, but the 'vocal' stem is near-silent --
    simulates Demucs correctly reporting no meaningful vocal content on an
    instrumental track."""
    return {
        "vocal": np.zeros_like(audio) + 1e-6,
        "drums": audio * 0.8,
        "bass": audio * 0.7,
        "instrumental": audio * 0.6,
    }


def make_sine_wav(path, duration_sec, freq=440.0, sr=CLAP_SR):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, sr)


@pytest.fixture
def repos(tmp_path):
    conn = init_db(tmp_path / "artifacts" / "sonic_explorer.db")
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=tmp_path / "artifacts")
    return song_repo, embedding_repo


@pytest.fixture
def curated_audio_with_songs(tmp_path, repos):
    """Pre-populates songs+segments in the DB (as if the sound-facet batch
    job already ran) -- run_batch_stem_embedding never creates songs itself,
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


def test_run_batch_stem_embedding_embeds_all_requested_facets(repos, curated_audio_with_songs):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    stem_facets = {"vocal": FakeStemFacet("vocal"), "drums": FakeStemFacet("drums")}

    run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo, stem_facets, separate_fn=fake_separate_stems,
    )

    song_a = song_repo.get_song_by_fma_track_id(1)
    for seg in song_a.segments:
        assert embedding_repo.status(seg.id, "vocal") == "done"
        assert embedding_repo.status(seg.id, "drums") == "done"
    assert embedding_repo.index_size("vocal") == embedding_repo.index_size("drums")


def test_run_batch_stem_embedding_skips_songs_not_in_db(repos, curated_audio_with_songs):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    manifest = pd.concat([manifest, pd.DataFrame([{"track_id": 999, "relative_path": "missing.wav"}])])
    stem_facets = {"vocal": FakeStemFacet("vocal")}

    failed = run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo, stem_facets, separate_fn=fake_separate_stems,
    )
    assert failed == []  # skipped, not failed -- track 999 was never in the DB


def test_run_batch_stem_embedding_is_idempotent(repos, curated_audio_with_songs):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    stem_facets = {"vocal": FakeStemFacet("vocal")}

    run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo, stem_facets, separate_fn=fake_separate_stems,
    )
    first_size = embedding_repo.index_size("vocal")

    call_count = {"n": 0}

    def counting_separate(audio, sr):
        call_count["n"] += 1
        return fake_separate_stems(audio, sr)

    run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo, stem_facets, separate_fn=counting_separate,
    )
    assert call_count["n"] == 0  # already fully embedded -- separation never re-runs
    assert embedding_repo.index_size("vocal") == first_size


def test_run_batch_stem_embedding_isolates_per_song_separation_failures(repos, curated_audio_with_songs):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    stem_facets = {"vocal": FakeStemFacet("vocal")}

    def flaky_separate(audio, sr):
        if len(audio) < CLAP_SR * 10:  # song 2 is 8s -- shorter than song 1's 12s
            raise RuntimeError("simulated separation failure")
        return fake_separate_stems(audio, sr)

    errors = []
    failed = run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo, stem_facets,
        separate_fn=flaky_separate, on_error=lambda track_id, exc: errors.append(track_id),
    )

    assert failed == [2]
    assert errors == [2]
    song_a = song_repo.get_song_by_fma_track_id(1)
    for seg in song_a.segments:
        assert embedding_repo.status(seg.id, "vocal") == "done"


def test_run_batch_stem_embedding_partial_completion_only_computes_missing_facets(repos, curated_audio_with_songs):
    """If vocal is already fully embedded but drums isn't, a re-run must
    still separate audio (drums needs it) while never re-embedding vocal."""
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    vocal_facet = FakeStemFacet("vocal")

    run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo, {"vocal": vocal_facet}, separate_fn=fake_separate_stems,
    )
    vocal_size = embedding_repo.index_size("vocal")

    vocal_facet2 = FakeStemFacet("vocal")
    drums_facet = FakeStemFacet("drums")
    run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo,
        {"vocal": vocal_facet2, "drums": drums_facet}, separate_fn=fake_separate_stems,
    )

    assert vocal_facet2.call_count == 0
    assert drums_facet.call_count > 0
    assert embedding_repo.index_size("vocal") == vocal_size
    assert embedding_repo.index_size("drums") == vocal_size


def test_run_batch_stem_embedding_checkpoints_and_leaves_no_orphaned_done_status(repos, tmp_path):
    """A mid-batch separation failure on one song must not affect the
    durability of already-checkpointed songs before it -- same invariant as
    embed_library.py's crash regression test, adapted for embed_stems.py's
    per-song error isolation (a bad file here is skipped and reported, not
    allowed to crash the whole batch, unlike the sound-facet pipeline)."""
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

    call_count = {"n": 0}

    def flaky_separate(audio, sr):
        call_count["n"] += 1
        if call_count["n"] == 3:  # song 3 -- processed after the checkpoint_every=2 flush
            raise RuntimeError("simulated crash")
        return fake_separate_stems(audio, sr)

    errors = []
    failed = run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo, {"vocal": FakeStemFacet("vocal")},
        checkpoint_every=2, separate_fn=flaky_separate,
        on_error=lambda track_id, exc: errors.append(track_id),
    )

    assert failed == [3]
    assert errors == [3]

    for track_id in (1, 2):
        song = song_repo.get_song_by_fma_track_id(track_id)
        for seg in song.segments:
            assert embedding_repo.status(seg.id, "vocal") == "done"
            embedding_repo.get_vector("vocal", seg.id)  # must not raise -- vector actually persisted

    song3 = song_repo.get_song_by_fma_track_id(3)
    for seg in song3.segments:
        assert embedding_repo.status(seg.id, "vocal") != "done"


def test_has_meaningful_energy_distinguishes_silence_from_signal():
    sr = CLAP_SR
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    silence = np.zeros_like(signal) + 1e-6

    assert _has_meaningful_energy(signal, signal) is True  # same energy as the "mix" -- clearly meaningful
    assert _has_meaningful_energy(silence, signal) is False  # near-zero next to a real mix -- not meaningful


def test_has_meaningful_energy_handles_empty_windows():
    empty = np.array([], dtype=np.float32)
    assert _has_meaningful_energy(empty, empty) is False


def test_run_batch_stem_embedding_marks_silent_stem_as_skipped_not_embedded(repos, curated_audio_with_songs):
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    vocal_facet = FakeStemFacet("vocal")

    run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo, {"vocal": vocal_facet},
        separate_fn=fake_separate_stems_silent_vocal,
    )

    assert vocal_facet.call_count == 0  # never embedded -- every segment's vocal stem was near-silent
    assert embedding_repo.index_size("vocal") == 0
    song_a = song_repo.get_song_by_fma_track_id(1)
    for seg in song_a.segments:
        assert embedding_repo.status(seg.id, "vocal") == "skipped"


def test_run_batch_stem_embedding_skipped_counts_as_finished_for_resumability(repos, curated_audio_with_songs):
    """A song whose vocal stem is entirely silent must not trigger a fresh
    Demucs separation on every future run -- 'skipped' has to satisfy the
    same per-song completion check 'done' does."""
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs

    run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo, {"vocal": FakeStemFacet("vocal")},
        separate_fn=fake_separate_stems_silent_vocal,
    )

    call_count = {"n": 0}

    def counting_separate(audio, sr):
        call_count["n"] += 1
        return fake_separate_stems_silent_vocal(audio, sr)

    run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo, {"vocal": FakeStemFacet("vocal")},
        separate_fn=counting_separate,
    )
    assert call_count["n"] == 0  # both songs already fully resolved (skipped) -- no re-separation


def test_run_batch_stem_embedding_only_embeds_facets_with_real_energy_within_same_song(repos, curated_audio_with_songs):
    """Within one song, a facet with real energy (drums) must still be
    embedded normally even though a sibling facet (vocal) is silent and
    skipped -- the energy gate is per-facet, not per-song."""
    song_repo, embedding_repo = repos
    audio_dir, manifest = curated_audio_with_songs
    vocal_facet = FakeStemFacet("vocal")
    drums_facet = FakeStemFacet("drums")

    run_batch_stem_embedding(
        manifest, audio_dir, song_repo, embedding_repo,
        {"vocal": vocal_facet, "drums": drums_facet}, separate_fn=fake_separate_stems_silent_vocal,
    )

    assert vocal_facet.call_count == 0
    assert drums_facet.call_count > 0
    song_a = song_repo.get_song_by_fma_track_id(1)
    for seg in song_a.segments:
        assert embedding_repo.status(seg.id, "vocal") == "skipped"
        assert embedding_repo.status(seg.id, "drums") == "done"
