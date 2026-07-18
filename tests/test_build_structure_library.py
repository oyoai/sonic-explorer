import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sonic_explorer.config import STRUCTURE_SR
from sonic_explorer.models import Song
from sonic_explorer.pipeline.build_structure_library import run_batch_structure
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.song_repository import SongRepository


def make_sine_wav(path, duration_sec=10.0, freq=440.0, sr=STRUCTURE_SR):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, sr)


@pytest.fixture
def song_repo(tmp_path):
    conn = init_db(tmp_path / "db.sqlite")
    return SongRepository(conn)


@pytest.fixture
def curated_audio(tmp_path, song_repo):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    make_sine_wav(audio_dir / "1.wav", freq=440.0)
    make_sine_wav(audio_dir / "2.wav", freq=880.0)

    song_repo.add_song(Song(filepath="x", fma_track_id=1, title="A", artist="A", genre_top="Rock", duration_sec=10.0))
    song_repo.add_song(Song(filepath="x", fma_track_id=2, title="B", artist="B", genre_top="Jazz", duration_sec=10.0))

    manifest = pd.DataFrame([
        {"track_id": 1, "relative_path": "1.wav"},
        {"track_id": 2, "relative_path": "2.wav"},
    ])
    return audio_dir, manifest


def test_run_batch_structure_creates_matrix_per_song(song_repo, curated_audio, tmp_path):
    audio_dir, manifest = curated_audio
    structure_dir = tmp_path / "structure"

    run_batch_structure(manifest, audio_dir, song_repo, structure_dir)

    song_a = song_repo.get_song_by_fma_track_id(1)
    song_b = song_repo.get_song_by_fma_track_id(2)
    assert (structure_dir / f"{song_a.id}.npy").exists()
    assert (structure_dir / f"{song_b.id}.npy").exists()

    matrix = np.load(structure_dir / f"{song_a.id}.npy")
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]


def test_run_batch_structure_skips_tracks_not_in_db(song_repo, curated_audio, tmp_path):
    audio_dir, manifest = curated_audio
    manifest = pd.concat([manifest, pd.DataFrame([{"track_id": 999, "relative_path": "missing.wav"}])])
    structure_dir = tmp_path / "structure"

    run_batch_structure(manifest, audio_dir, song_repo, structure_dir)  # should not raise on track 999

    assert len(list(structure_dir.glob("*.npy"))) == 2


def test_run_batch_structure_is_idempotent(song_repo, curated_audio, tmp_path, monkeypatch):
    audio_dir, manifest = curated_audio
    structure_dir = tmp_path / "structure"

    run_batch_structure(manifest, audio_dir, song_repo, structure_dir)

    import librosa

    def boom(*args, **kwargs):
        raise AssertionError("librosa.load should not be called again for already-computed songs")

    monkeypatch.setattr(librosa, "load", boom)

    run_batch_structure(manifest, audio_dir, song_repo, structure_dir)  # should not touch librosa at all
