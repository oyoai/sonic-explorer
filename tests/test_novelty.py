import numpy as np
import pytest

from sonic_explorer.config import CLAP_SR
from sonic_explorer.facets.novelty import compute_novelty_curve, compute_structural_confidence, find_structural_peaks


def make_sine(duration_sec=10.0, freq=440.0, sr=CLAP_SR):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def chroma_for(audio, sr):
    import librosa

    return librosa.feature.chroma_cqt(y=audio, sr=sr)


def test_novelty_curve_shape_and_range():
    audio = make_sine(duration_sec=15.0)
    chroma = chroma_for(audio, CLAP_SR)
    curve = compute_novelty_curve(chroma, duration_sec=15.0)

    assert curve.shape == (chroma.shape[-1],)
    assert curve.min() >= 0.0
    assert curve.max() <= 1.0 + 1e-6


def test_novelty_curve_handles_tiny_input():
    curve = compute_novelty_curve(np.zeros((12, 2)), duration_sec=1.0)
    assert curve.shape == (2,)
    assert np.all(curve == 0.0)


def test_pure_tone_has_no_structural_peaks():
    """A single sustained tone has no genuine section boundaries -- confirmed
    against real behavior (0 peaks) during prototyping."""
    audio = make_sine(duration_sec=20.0, freq=440.0)
    chroma = chroma_for(audio, CLAP_SR)
    curve = compute_novelty_curve(chroma, duration_sec=20.0)
    peaks = find_structural_peaks(curve)

    assert len(peaks) == 0


def test_two_distinct_halves_produces_one_peak_near_midpoint():
    """Sanity check for both the kernel's adaptive sizing (this exercises the
    framewise-fallback frame rate, not beat-synced) and its actual boundary-
    detection accuracy -- confirmed during prototyping to land within ~0.1s of
    the true 15.0s midpoint."""
    sr = CLAP_SR
    half_duration = 15.0
    t = np.linspace(0, half_duration, int(half_duration * sr), endpoint=False)
    first_half = (0.2 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32)  # C4
    second_half = (0.2 * np.sin(2 * np.pi * 369.99 * t)).astype(np.float32)  # F#4
    audio = np.concatenate([first_half, second_half])
    duration_sec = len(audio) / sr

    chroma = chroma_for(audio, sr)
    curve = compute_novelty_curve(chroma, duration_sec)
    peaks = find_structural_peaks(curve)

    assert len(peaks) == 1
    frame_times = np.linspace(0, duration_sec, len(curve))
    assert frame_times[peaks[0]] == pytest.approx(half_duration, abs=1.0)


def test_find_structural_peaks_respects_prominence_threshold():
    curve = np.array([0.0, 0.05, 0.0, 0.9, 0.0, 0.05, 0.0], dtype=np.float32)
    strict_peaks = find_structural_peaks(curve, min_prominence=0.5)
    loose_peaks = find_structural_peaks(curve, min_prominence=0.01)

    assert len(strict_peaks) == 1  # only the big peak clears a high bar
    assert len(loose_peaks) >= len(strict_peaks)  # a low bar admits at least as many


def test_find_structural_peaks_handles_tiny_input():
    assert len(find_structural_peaks(np.array([0.5]))) == 0
    assert len(find_structural_peaks(np.array([]))) == 0


def test_structural_confidence_is_zero_for_flat_curve():
    flat = np.full(20, 0.5, dtype=np.float32)
    assert compute_structural_confidence(flat) == pytest.approx(0.0)


def test_structural_confidence_is_positive_for_varied_curve():
    varied = np.array([0.0, 0.1, 0.9, 0.1, 0.0, 0.8, 0.1], dtype=np.float32)
    assert compute_structural_confidence(varied) > 0.0


def test_structural_confidence_handles_empty_curve():
    assert compute_structural_confidence(np.array([])) == 0.0
