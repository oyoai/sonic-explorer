"""Cheap, display-only waveform envelopes for Approach's step-by-step
visuals -- NOT analysis-grade audio (that's CLAP/chroma/AST elsewhere at
their real sample rates). Loaded at a low sample rate purely to get a fast,
real amplitude shape to plot; this module never feeds anything back into a
similarity computation.

Plain Python, no Streamlit import -- see spec 8.3's core/interface
separation, same discipline as network_graph.py/taste_map.py."""

import numpy as np

PREVIEW_SR = 4000  # low -- this is for a visual shape, not audio analysis
DEFAULT_N_POINTS = 300


def waveform_envelope(path, n_points: int = DEFAULT_N_POINTS) -> np.ndarray:
    """Downsampled amplitude envelope (max of abs() per block) -- a real
    waveform shape cheap enough to compute per page render, not a placeholder
    sine wave or synthetic shape."""
    import librosa

    audio, _ = librosa.load(str(path), sr=PREVIEW_SR, mono=True)
    if len(audio) == 0:
        return np.zeros(n_points)

    block = max(1, len(audio) // n_points)
    trimmed = audio[: block * n_points]
    return np.abs(trimmed).reshape(n_points, block).max(axis=1)
