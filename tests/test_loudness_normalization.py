import numpy as np
import pytest

from sonic_explorer.evaluation.loudness_normalization import normalize_peak


def test_normalize_peak_scales_max_sample_to_target_amplitude():
    audio = np.array([0.1, -0.4, 0.2, -0.05], dtype=np.float32)
    normalized = normalize_peak(audio, target_db=-1.0)

    expected_peak = 10.0 ** (-1.0 / 20.0)
    assert np.max(np.abs(normalized)) == pytest.approx(expected_peak, rel=1e-5)


def test_normalize_peak_preserves_relative_shape():
    """Peak normalization is a single scalar multiply -- every sample keeps
    its ratio to every other sample, only the overall scale changes."""
    audio = np.array([0.1, -0.4, 0.2, -0.05], dtype=np.float32)
    normalized = normalize_peak(audio)

    ratio = normalized[1] / normalized[0]
    original_ratio = audio[1] / audio[0]
    assert ratio == pytest.approx(original_ratio, rel=1e-5)


def test_normalize_peak_handles_silence_without_dividing_by_zero():
    silence = np.zeros(10, dtype=np.float32)
    normalized = normalize_peak(silence)

    assert np.all(normalized == 0.0)
    assert np.all(np.isfinite(normalized))


def test_normalize_peak_makes_differently_gained_versions_of_the_same_signal_converge():
    """The actual property a perturbation test relies on: two clips that are
    the SAME underlying signal at different raw gain levels must land at
    (near enough) the same normalized audio -- otherwise a comparison after
    normalization would still be partly comparing loudness, not just
    whatever the real perturbation changed."""
    base = np.array([0.05, -0.2, 0.15, -0.1, 0.3], dtype=np.float32)
    quiet = base * 0.2
    loud = base * 3.0

    norm_quiet = normalize_peak(quiet)
    norm_loud = normalize_peak(loud)

    np.testing.assert_allclose(norm_quiet, norm_loud, rtol=1e-5)


def test_normalize_peak_default_target_is_minus_one_db():
    audio = np.array([1.0, -0.5], dtype=np.float32)
    normalized = normalize_peak(audio)

    assert np.max(np.abs(normalized)) == pytest.approx(10.0 ** (-1.0 / 20.0), rel=1e-5)
