"""Song fingerprints: small per-song thumbnail arrays used as an album-art
fallback and as visual identity throughout the app (spec 2.6). Structure and
sound fingerprints are Core; harmony/composite are Strong, now that the
harmony facet exists.

structure_fingerprint() is a pure downsample of the already-computed self-
similarity matrix -- no audio/librosa dependency, safe to compute on the fly
anywhere, including the deployed app. sound_fingerprint()/harmony_fingerprint()
need the raw waveform (mel-spectrogram / chroma-gram), so both are precomputed
in the batch pipeline where librosa is already in use (see
pipeline/build_structure_library.py) and persisted -- the deployed app only
ever reads them back, consistent with the rest of the architecture (deployed
app never runs audio processing itself).

structure_fingerprint()/sound_fingerprint() return a size x size float32 array
normalized to [0, 1], ready for a perceptually-uniform colormap
(viridis/magma/plasma) at display time -- see streamlit_app/components/plotting.py.
harmony_fingerprint() returns a (12, size) strip instead of a square -- a
chroma-gram only ever has 12 meaningful rows (pitch classes), so squaring it
would blur a real distinction into an arbitrary one. composite_fingerprint()
reconciles the shapes for the RGB overlay.
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


HARMONY_FINGERPRINT_ROWS = 12  # one row per pitch class -- chroma has no more real resolution than this


def harmony_fingerprint(audio: np.ndarray, sr: int, size: int = FINGERPRINT_SIZE) -> np.ndarray:
    """Chroma-gram strip: 12 pitch-class rows x `size` downsampled time steps.
    Reuses the same whole-song audio load as sound_fingerprint/structure
    computation (see pipeline/build_structure_library.py) -- no extra I/O."""
    import librosa

    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    if chroma.shape[1] == 0:
        return np.zeros((HARMONY_FINGERPRINT_ROWS, size), dtype=np.float32)
    return _normalize(_block_average_2d(chroma, HARMONY_FINGERPRINT_ROWS, size))


def composite_fingerprint(structure_fp: np.ndarray, sound_fp: np.ndarray, harmony_fp: np.ndarray) -> np.ndarray:
    """RGB-channel overlay of the three facet fingerprints (spec 2.6): structure
    -> red, harmony -> green, sound -> blue, each already normalized to [0, 1]
    grayscale intensity. Where all three agree the composite reads bright/
    white; where they diverge, distinct color casts appear -- two songs similar
    in structure but not harmony would show similarly-patterned but
    differently-colored composites.

    harmony_fp's native (12, size) strip is stretched to the square (size,
    size) grid the other two already share, purely so the three channels align
    pixel-for-pixel here -- the standalone harmony fingerprint shown elsewhere
    stays the honest 12-row strip."""
    size = structure_fp.shape[0]
    harmony_square = _block_average_2d(harmony_fp, size, size)
    return np.stack([structure_fp, harmony_square, sound_fp], axis=-1).astype(np.float32)
