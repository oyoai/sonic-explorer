"""Song fingerprints: small per-song thumbnail arrays used as an album-art
fallback and as visual identity throughout the app (spec 2.6). Structure and
sound fingerprints are Core; harmony/composite are Strong (once the harmony
facet exists) -- only structure + sound are built here.

structure_fingerprint() is a pure downsample of the already-computed self-
similarity matrix -- no audio/librosa dependency, safe to compute on the fly
anywhere, including the deployed app. sound_fingerprint() needs the raw
waveform (mel-spectrogram), so it's precomputed in the batch pipeline where
librosa is already in use (see pipeline/build_structure_library.py) and
persisted -- the deployed app only ever reads it back, consistent with the
rest of the architecture (deployed app never runs audio processing itself).

Both return a size x size float32 array normalized to [0, 1], ready for a
perceptually-uniform colormap (viridis/magma/plasma) at display time -- see
streamlit_app/components/plotting.py.
"""

import numpy as np

FINGERPRINT_SIZE = 32


def _block_average_2d(source: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    """Downsamples a 2D array to out_rows x out_cols by averaging each block of
    the input that maps to one output pixel. Handles input dims that don't evenly
    divide the output size (typical here -- SSMs/spectrograms are all different
    lengths)."""
    n_rows, n_cols = source.shape
    row_edges = np.linspace(0, n_rows, out_rows + 1).astype(int)
    col_edges = np.linspace(0, n_cols, out_cols + 1).astype(int)
    thumb = np.zeros((out_rows, out_cols), dtype=np.float32)
    for i in range(out_rows):
        r0, r1 = row_edges[i], max(row_edges[i] + 1, row_edges[i + 1])
        for j in range(out_cols):
            c0, c1 = col_edges[j], max(col_edges[j] + 1, col_edges[j + 1])
            thumb[i, j] = source[r0:r1, c0:c1].mean()
    return thumb


def _normalize(thumb: np.ndarray) -> np.ndarray:
    lo, hi = float(thumb.min()), float(thumb.max())
    return (thumb - lo) / (hi - lo) if hi > lo else np.zeros_like(thumb)


def structure_fingerprint(matrix: np.ndarray, size: int = FINGERPRINT_SIZE) -> np.ndarray:
    """Downsampled tile of the self-similarity matrix -- reuses the existing
    structure computation, just rendered small."""
    if matrix.shape[0] == 0:
        return np.zeros((size, size), dtype=np.float32)
    return _normalize(_block_average_2d(matrix, size, size))


def sound_fingerprint(audio: np.ndarray, sr: int, size: int = FINGERPRINT_SIZE) -> np.ndarray:
    """Mel-spectrogram thumbnail -- captures acoustic texture visually, doesn't
    require CLAP, cheap once audio is loaded (the batch pipeline already loads it
    for structure computation, so this reuses that same load, no extra I/O)."""
    import librosa

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=size)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return _normalize(_block_average_2d(mel_db, size, size))
