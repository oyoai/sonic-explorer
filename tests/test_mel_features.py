import numpy as np

from sonic_explorer.analysis.mel_features import N_FRAMES, N_MELS, extract_mel_spectrogram


def _sine_wave(duration_sec: float, sr: int, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_extract_mel_spectrogram_returns_fixed_shape_for_nominal_length():
    audio = _sine_wave(30.0, sr=48000)
    mel = extract_mel_spectrogram(audio, sr=48000)
    assert mel.shape == (N_MELS, N_FRAMES)


def test_extract_mel_spectrogram_pads_short_clips():
    audio = _sine_wave(5.0, sr=48000)  # much shorter than the nominal 30s
    mel = extract_mel_spectrogram(audio, sr=48000)
    assert mel.shape == (N_MELS, N_FRAMES)


def test_extract_mel_spectrogram_crops_long_clips():
    audio = _sine_wave(45.0, sr=48000)  # longer than the nominal 30s
    mel = extract_mel_spectrogram(audio, sr=48000)
    assert mel.shape == (N_MELS, N_FRAMES)


def test_extract_mel_spectrogram_output_is_roughly_normalized():
    audio = _sine_wave(30.0, sr=48000)
    mel = extract_mel_spectrogram(audio, sr=48000)
    # log-mel dB is typically [-80, 0] -> normalized to roughly [-1, 1]
    assert mel.min() >= -3.0
    assert mel.max() <= 3.0
