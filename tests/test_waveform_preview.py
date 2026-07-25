import numpy as np
import soundfile as sf

from sonic_explorer.analysis.waveform_preview import PREVIEW_SR, waveform_envelope


def make_sine_wav(path, duration_sec, freq=440.0, sr=PREVIEW_SR):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), audio, sr)


def test_waveform_envelope_returns_requested_length(tmp_path):
    path = tmp_path / "tone.wav"
    make_sine_wav(path, duration_sec=5.0)

    envelope = waveform_envelope(path, n_points=100)

    assert envelope.shape == (100,)


def test_waveform_envelope_nonzero_for_real_tone(tmp_path):
    path = tmp_path / "tone.wav"
    make_sine_wav(path, duration_sec=3.0, freq=440.0)

    envelope = waveform_envelope(path, n_points=50)

    assert envelope.max() > 0
    assert (envelope >= 0).all()


def test_waveform_envelope_silence_is_all_zero(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros(PREVIEW_SR * 2), PREVIEW_SR)

    envelope = waveform_envelope(path, n_points=50)

    assert envelope.max() == 0
