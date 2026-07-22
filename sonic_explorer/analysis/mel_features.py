"""Log-mel spectrogram extraction for the genre-CNN baseline (analysis/
genre_cnn.py) -- split into its own module specifically so it stays
importable without torch (librosa is already a dev/colab dependency,
unlike torch which isn't installed in the main CI environment). A CNN
needs a fixed input shape; clip lengths vary slightly around the nominal
30s, so this crops/pads rather than assuming every clip is exactly the
same length."""

import numpy as np

N_MELS = 128
N_FRAMES = 640  # ~30s at CLAP_SR=48000 with hop_length=2048


def extract_mel_spectrogram(
    audio: np.ndarray, sr: int, n_mels: int = N_MELS, n_frames: int = N_FRAMES
) -> np.ndarray:
    """Returns a (n_mels, n_frames) float32 array, roughly normalized to
    [-1, 1] (log-mel dB values are typically in [-80, 0])."""
    import librosa

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=2048)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape[1] < n_frames:
        pad = n_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant", constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :n_frames]

    return (log_mel / 40.0 + 1.0).astype(np.float32)
