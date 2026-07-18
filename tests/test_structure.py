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
