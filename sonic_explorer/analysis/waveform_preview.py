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


def extract_window_clip(path, start_sec: float, duration_sec: float, sr: int = 22050) -> bytes:
    """Real, playable WAV bytes for one exact time-range clip of a song --
    Approach's windowing step needs actual separate playable clips per
    window, not just a visual highlight overlaid on the continuous
    full-song player. Unlike waveform_envelope above (a cheap shape for
    plotting, PREVIEW_SR=4000), this is real audio meant to be listened to,
    so it loads at a normal listening sample rate -- still not the
    analysis-grade CLAP_SR, since nothing here feeds a similarity
    computation.

    Deliberately loads the WHOLE file via plain librosa.load(path, sr=sr,
    mono=True) and slices the resulting array in-memory, rather than
    librosa's own offset/duration load params -- those force a different
    internal decode path (soundfile can't seek efficiently within a
    compressed MP3 stream, so librosa falls back to audioread) than the
    plain full-file load every other caller on this page already uses
    successfully. That fallback path broke on a real Streamlit Cloud
    deploy running Python 3.14 (ImportError inside librosa's own load(),
    redacted by Streamlit before reaching this repo's logs) -- Python
    3.13+ removed several stdlib modules audioread's older fallback chain
    can still reach for on some formats, and this project already carries
    standard-aifc/standard-sunau as an unpinned, transitively-resolved
    workaround for exactly that gap (see librosa's own DeprecationWarning
    in this repo's test output) -- not a guarantee that resolves identically
    on every fresh Python version. Loading the whole ~30s clip 4x for 4
    windows costs a few hundred ms, not a real performance concern here."""
    import io

    import librosa
    import soundfile as sf

    audio, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    start_idx = int(start_sec * loaded_sr)
    end_idx = int((start_sec + duration_sec) * loaded_sr)
    clip = audio[start_idx:end_idx]

    buf = io.BytesIO()
    sf.write(buf, clip, loaded_sr, format="WAV")
    return buf.getvalue()
