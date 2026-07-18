import numpy as np

from sonic_explorer.config import CLAP_SR
from sonic_explorer.facets.structure import compute_self_similarity_matrix


def make_sine(duration_sec=10.0, freq=440.0, sr=CLAP_SR):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_self_similarity_matrix_is_square_and_symmetric():
    audio = make_sine()
    matrix = compute_self_similarity_matrix(audio, CLAP_SR)

    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert np.allclose(matrix, matrix.T, atol=1e-5)


def test_self_similarity_matrix_has_finite_values():
    audio = make_sine()
    matrix = compute_self_similarity_matrix(audio, CLAP_SR)

    assert np.all(np.isfinite(matrix))
    assert matrix.shape[0] > 1


def test_self_similarity_matrix_falls_back_when_synced_chroma_too_short(monkeypatch):
    """Regression test for a real crash found on real FMA data: some tracks (near-
    silent intros, sparse/ambient sections) produce a beat-synced chroma with too
    few frames for librosa.segment.recurrence_matrix's default width, which raised
    a ParameterError outright. Forces that exact condition deterministically rather
    than hunting for a specific real audio file that triggers it."""
    import librosa

    audio = make_sine(duration_sec=5.0)

    monkeypatch.setattr(librosa.beat, "beat_track", lambda y, sr: (120.0, np.array([0, 1])))
    monkeypatch.setattr(librosa.util, "sync", lambda chroma, beats: chroma[:, :2])  # too-short synced result

    matrix = compute_self_similarity_matrix(audio, CLAP_SR)

    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] > 2  # used the framewise fallback, not the 2-frame synced chroma
    assert np.all(np.isfinite(matrix))
