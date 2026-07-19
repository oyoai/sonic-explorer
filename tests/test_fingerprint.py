import numpy as np

from sonic_explorer.config import CLAP_SR
from sonic_explorer.facets.fingerprint import FINGERPRINT_SIZE, sound_fingerprint, structure_fingerprint


def make_sine(duration_sec=10.0, freq=440.0, sr=CLAP_SR):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_structure_fingerprint_shape_and_range():
    matrix = np.random.default_rng(0).random((50, 50)).astype(np.float32)
    fp = structure_fingerprint(matrix)

    assert fp.shape == (FINGERPRINT_SIZE, FINGERPRINT_SIZE)
    assert fp.min() >= 0.0
    assert fp.max() <= 1.0 + 1e-6


def test_structure_fingerprint_handles_matrix_smaller_than_target_size():
    matrix = np.random.default_rng(0).random((10, 10)).astype(np.float32)
    fp = structure_fingerprint(matrix, size=32)

    assert fp.shape == (32, 32)
    assert np.all(np.isfinite(fp))


def test_structure_fingerprint_handles_empty_matrix():
    fp = structure_fingerprint(np.zeros((0, 0), dtype=np.float32))
    assert fp.shape == (FINGERPRINT_SIZE, FINGERPRINT_SIZE)
    assert np.all(fp == 0.0)


def test_structure_fingerprint_constant_input_is_all_zero():
    matrix = np.full((20, 20), 5.0, dtype=np.float32)
    fp = structure_fingerprint(matrix)
    assert np.all(fp == 0.0)


def test_structure_fingerprint_preserves_block_structure():
    """Two clearly different halves (top-left bright, bottom-right dim) should
    downsample into a fingerprint that still shows that same pattern -- a
    correctness check on the block-averaging, not just shape/range."""
    matrix = np.zeros((40, 40), dtype=np.float32)
    matrix[:20, :20] = 1.0
    fp = structure_fingerprint(matrix, size=8)

    assert fp[0, 0] > fp[-1, -1]


def test_sound_fingerprint_shape_and_range():
    audio = make_sine()
    fp = sound_fingerprint(audio, CLAP_SR)

    assert fp.shape == (FINGERPRINT_SIZE, FINGERPRINT_SIZE)
    assert fp.min() >= 0.0
    assert fp.max() <= 1.0 + 1e-6
    assert np.all(np.isfinite(fp))


def test_sound_fingerprint_custom_size():
    audio = make_sine(duration_sec=5.0)
    fp = sound_fingerprint(audio, CLAP_SR, size=16)
    assert fp.shape == (16, 16)
