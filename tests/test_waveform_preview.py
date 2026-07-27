import io

import numpy as np
import soundfile as sf

from sonic_explorer.analysis.waveform_preview import PREVIEW_SR, extract_window_clip, waveform_envelope


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


def test_extract_window_clip_returns_valid_wav_bytes_of_the_requested_duration(tmp_path):
    path = tmp_path / "tone.wav"
    make_sine_wav(path, duration_sec=10.0, freq=440.0, sr=22050)

    clip_bytes = extract_window_clip(path, start_sec=2.0, duration_sec=5.0, sr=22050)

    assert isinstance(clip_bytes, bytes)
    audio, sr = sf.read(io.BytesIO(clip_bytes))
    assert sr == 22050
    assert abs(len(audio) / sr - 5.0) < 0.05  # real ~5s clip, not the whole 10s file


def test_extract_window_clip_offset_actually_moves_the_window(tmp_path):
    """A clip starting partway through a tone that fades in must differ from
    a clip starting at 0 -- confirms `offset` is really slicing, not just
    truncating duration from the start every time."""
    path = tmp_path / "fade.wav"
    sr = 22050
    duration_sec = 10.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    fade_in = np.clip(t / duration_sec, 0, 1)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t) * fade_in
    sf.write(str(path), audio, sr)

    early = extract_window_clip(path, start_sec=0.0, duration_sec=1.0, sr=sr)
    late = extract_window_clip(path, start_sec=8.0, duration_sec=1.0, sr=sr)

    early_audio, _ = sf.read(io.BytesIO(early))
    late_audio, _ = sf.read(io.BytesIO(late))
    assert np.abs(late_audio).mean() > np.abs(early_audio).mean() * 2  # late window is much louder (faded in)
