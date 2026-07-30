import numpy as np
import pytest

from sonic_explorer.evaluation.clap_gain_sensitivity import apply_gain_db, measure_gain_sensitivity


def test_apply_gain_db_scales_amplitude():
    audio = np.array([0.1, -0.1, 0.05], dtype=np.float32)
    boosted = apply_gain_db(audio, 6.0)
    np.testing.assert_allclose(boosted, audio * (10.0 ** (6.0 / 20.0)), rtol=1e-5)


def test_apply_gain_db_cuts_amplitude_for_negative_db():
    audio = np.array([0.4, -0.4], dtype=np.float32)
    cut = apply_gain_db(audio, -6.0)
    assert np.all(np.abs(cut) < np.abs(audio))


def test_apply_gain_db_clips_to_valid_sample_range():
    """Without clipping, "loudness perturbation" at large gain_db would
    silently become "loudness perturbation plus hard-clipping distortion" --
    a real waveform-shape change, not a pure gain change, which would
    confound the measurement this module exists to isolate."""
    audio = np.array([0.9, -0.9, 0.5], dtype=np.float32)
    boosted = apply_gain_db(audio, 12.0)
    assert np.all(boosted <= 1.0) and np.all(boosted >= -1.0)


class _GainInvariantFacet:
    """Stand-in for "this embedding genuinely doesn't react to loudness" --
    always the same direction regardless of input amplitude."""

    def embed_batch(self, audio_windows, sr, batch_size=8):
        return np.stack([np.array([1.0, 0.0, 0.0], dtype=np.float32) for _ in audio_windows])


class _GainSensitiveFacet:
    """Stand-in for "this embedding's DIRECTION shifts with loudness," not
    just magnitude -- a pure magnitude-only embedding (e.g. [amplitude, 0,
    0]) would be a scalar multiple of itself at every gain level, and
    cosine similarity is scale-invariant by definition, so that alone
    wouldn't actually exercise drift detection. Fixing two components and
    only varying one changes the vector's ANGLE as amplitude changes,
    which is what a real gain-sensitive embedding would need to do to show
    up as reduced cosine similarity at all."""

    def embed_batch(self, audio_windows, sr, batch_size=8):
        return np.stack([
            np.array([float(np.max(np.abs(w))), 0.5, 0.1], dtype=np.float32) for w in audio_windows
        ])


def test_measure_gain_sensitivity_reports_no_drift_for_an_invariant_embedding():
    windows = [np.array([0.1, 0.2, -0.1, 0.05], dtype=np.float32) for _ in range(3)]
    results = measure_gain_sensitivity(windows, gain_levels_db=[-6.0, 6.0], sr=48000, facet=_GainInvariantFacet())

    assert [r.gain_db for r in results] == [-6.0, 6.0]
    for r in results:
        assert len(r.cosine_similarities) == 3
        assert r.mean_similarity == pytest.approx(1.0, abs=1e-6)
        assert r.mean_drift == pytest.approx(0.0, abs=1e-6)


def test_measure_gain_sensitivity_detects_drift_for_a_direction_shifting_embedding():
    windows = [np.array([0.1, 0.2, -0.1, 0.05], dtype=np.float32) for _ in range(3)]
    results = measure_gain_sensitivity(windows, gain_levels_db=[-6.0, 12.0], sr=48000, facet=_GainSensitiveFacet())

    for r in results:
        assert r.mean_similarity < 1.0
        assert r.mean_drift > 0.0
