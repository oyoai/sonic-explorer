"""Structure: a song's self-similarity matrix (verse/chorus shape) -- promoted from
the beat-synced chroma recurrence matrix in notebooks/audio_deep_dive.ipynb (cell 21).
Powers Song X-Ray directly. Core's UI does not do structure-based retrieval (see the
spec's Core/Strong split -- Moment Matcher is sound-only in Core), so this is a
song-level artifact computed directly, not routed through the Facet/FacetRegistry
retrieval path the way SoundFacet is."""

import numpy as np


def compute_self_similarity_matrix(audio: np.ndarray, sr: int) -> np.ndarray:
    """Beat-synced chroma recurrence matrix. Falls back to unsynced chroma if beat
    tracking finds fewer than 2 beats (e.g. a near-silent or non-rhythmic clip)."""
    import librosa

    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    chroma_sync = librosa.util.sync(chroma, beat_frames) if len(beat_frames) >= 2 else chroma
    return librosa.segment.recurrence_matrix(chroma_sync, mode="affinity", sym=True).astype(np.float32)
