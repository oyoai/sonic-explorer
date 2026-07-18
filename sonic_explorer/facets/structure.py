"""Structure: a song's self-similarity matrix (verse/chorus shape) -- promoted from
the beat-synced chroma recurrence matrix in notebooks/audio_deep_dive.ipynb (cell 21).
Powers Song X-Ray directly. Core's UI does not do structure-based retrieval (see the
spec's Core/Strong split -- Moment Matcher is sound-only in Core), so this is a
song-level artifact computed directly, not routed through the Facet/FacetRegistry
retrieval path the way SoundFacet is."""

import numpy as np

# librosa.segment.recurrence_matrix's default width needs (n_frames - 1) // 2 >= 1,
# i.e. at least 3 frames -- a real failure mode on real FMA data: some tracks (near-
# silent intros, sparse/ambient sections, non-rhythmic clips) produce a beat-synced
# chroma with only 1-2 frames, which crashes recurrence_matrix outright.
MIN_FRAMES_FOR_RECURRENCE = 3


def compute_self_similarity_matrix(audio: np.ndarray, sr: int) -> np.ndarray:
    """Beat-synced chroma recurrence matrix. Falls back to unsynced (framewise)
    chroma if beat tracking finds too few beats to produce enough synced frames --
    framewise chroma has orders of magnitude more frames, so this is effectively
    always safe for any clip longer than a second or so."""
    import librosa

    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)

    chroma_sync = chroma
    if len(beat_frames) >= 2:
        synced = librosa.util.sync(chroma, beat_frames)
        if synced.shape[-1] >= MIN_FRAMES_FOR_RECURRENCE:
            chroma_sync = synced

    return librosa.segment.recurrence_matrix(chroma_sync, mode="affinity", sym=True).astype(np.float32)
