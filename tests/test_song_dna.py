import numpy as np

from sonic_explorer.config import CLAP_SR
from sonic_explorer.facets.song_dna import compute_raw_song_dna


def make_sine(duration_sec=10.0, freq=440.0, sr=CLAP_SR):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_compute_raw_song_dna_returns_all_expected_keys():
    audio = make_sine()
    dna = compute_raw_song_dna(audio, CLAP_SR)

    assert set(dna.keys()) == {"tempo_bpm", "energy", "brightness", "harmonic_complexity", "rhythmic_density"}
    assert all(isinstance(v, float) for v in dna.values())


def test_compute_raw_song_dna_values_are_finite_and_non_negative():
    audio = make_sine()
    dna = compute_raw_song_dna(audio, CLAP_SR)

    for key, value in dna.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"
        assert value >= 0.0, f"{key} is negative: {value}"


def test_harmonic_complexity_is_bounded_zero_to_one():
    """A single pure tone concentrates all chroma energy in one pitch class --
    minimal entropy, so harmonic_complexity should be low (near 0), and always
    within [0, 1] since it's normalized by max possible entropy."""
    audio = make_sine(duration_sec=10.0, freq=440.0)
    dna = compute_raw_song_dna(audio, CLAP_SR)

    assert 0.0 <= dna["harmonic_complexity"] <= 1.0
    assert dna["harmonic_complexity"] < 0.5  # a pure tone is tonally simple


def test_louder_audio_has_higher_energy():
    quiet = 0.05 * np.sin(2 * np.pi * 440 * np.linspace(0, 5, 5 * CLAP_SR, endpoint=False)).astype(np.float32)
    loud = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 5, 5 * CLAP_SR, endpoint=False)).astype(np.float32)

    dna_quiet = compute_raw_song_dna(quiet, CLAP_SR)
    dna_loud = compute_raw_song_dna(loud, CLAP_SR)

    assert dna_loud["energy"] > dna_quiet["energy"]
