"""Structure: a song's self-similarity matrix (verse/chorus shape) and a compact
segmented timeline (which stretches of the song sound alike) -- promoted from the
beat-synced chroma recurrence matrix (audio_deep_dive.ipynb cell 21) and the
clustering-into-a-template idea (cell 16 -- DBSCAN there; here, agglomerative
clustering with a temporal-adjacency constraint, for a reason found while
building this: plain per-frame K-Means (tried first) flips labels constantly even
within one perceptual section -- confirmed on real songs, 18-46 label changes per
~30s clip. A minimum-duration merge pass tried to paper over that and instead
collapsed 61% of the library down to a single undifferentiated segment, since
merges just kept propagating one label across the whole song. Constraining
clustering to only ever merge temporally-adjacent frames fixes this at the root:
each of the k clusters is *guaranteed* to be one contiguous interval by
construction (no adjacency, no merge), so the result is always exactly k
clean segments -- confirmed on the same real songs.

Powers Song X-Ray: the timeline is the primary, non-technical view (a color bar
where matching colors = similar-sounding sections); the matrix is the secondary
"technical detail" view underneath. Core's UI does not do structure-based
retrieval (see the spec's Core/Strong split), so none of this routes through the
Facet/FacetRegistry retrieval path the way SoundFacet does -- these are song-level
artifacts computed and cached directly.

Not every song has clear verse/chorus repetition -- ambient/through-composed
tracks genuinely don't, and the agglomerative clustering above will still
produce *something* regardless (it has no notion of "there's no real structure
here"). facets/novelty.py's checkerboard-kernel novelty curve is the separate
signal that decides whether to trust and show those segments, or show the
honest continuous curve instead -- see has_clear_structure below.
"""

from dataclasses import dataclass

import numpy as np

from sonic_explorer.facets.novelty import compute_novelty_curve, compute_structural_confidence, find_structural_peaks

# librosa.segment.recurrence_matrix's default width needs (n_frames - 1) // 2 >= 1,
# i.e. at least 3 frames -- a real failure mode on real FMA data: some tracks (near-
# silent intros, sparse/ambient sections, non-rhythmic clips) produce a beat-synced
# chroma with only 1-2 frames, which crashes recurrence_matrix outright.
MIN_FRAMES_FOR_RECURRENCE = 3

DEFAULT_TIMELINE_CLUSTERS = 6

# Safety net, not the primary mechanism (that's the connectivity constraint above)
# -- a cluster can still legitimately be very short (e.g. a one-beat transition),
# so this cleans up the rare edge case rather than doing the heavy lifting.
MIN_SEGMENT_SEC = 3.0


@dataclass
class StructureTimeline:
    """What EmbeddingRepository.get_structure_timeline() reads back from disk --
    the timeline segments plus the sound/harmony fingerprints, which piggyback
    on the same .npz file since all are computed from the same audio load in
    the batch pipeline (see pipeline/build_structure_library.py). The matrix
    itself is a separate file, read via get_structure_matrix()."""

    segment_starts: np.ndarray
    segment_ends: np.ndarray
    segment_labels: np.ndarray
    sound_fingerprint: np.ndarray | None = None
    harmony_fingerprint: np.ndarray | None = None
    novelty_curve: np.ndarray | None = None
    novelty_times: np.ndarray | None = None
    has_clear_structure: bool = True
    structural_confidence: float | None = None


@dataclass
class StructureAnalysis:
    matrix: np.ndarray  # self-similarity matrix, frames x frames (diagonal deliberately zeroed, see analyze_structure)
    segment_starts: np.ndarray  # start_sec per contiguous timeline segment
    segment_ends: np.ndarray  # end_sec per contiguous timeline segment
    segment_labels: np.ndarray  # cluster label (int) per contiguous timeline segment
    novelty_curve: np.ndarray  # one value per chroma_sync frame, see facets/novelty.py
    novelty_times: np.ndarray  # start_sec per novelty_curve entry
    has_clear_structure: bool  # False when no novelty peak clears the confidence threshold
    structural_confidence: float  # continuous flatness measure -- see facets/novelty.py


def _merge_short_segments(starts: list, ends: list, labels: list, min_duration: float) -> tuple[list, list, list]:
    """Merges any run shorter than min_duration into an adjacent run (extending
    that neighbor's boundary, adopting its label) -- repeats until every run meets
    the minimum or only one remains. Deterministic: merges into the previous run,
    or the next run if this is the very first segment."""
    starts, ends, labels = list(starts), list(ends), list(labels)
    changed = True
    while changed and len(starts) > 1:
        changed = False
        for i in range(len(starts)):
            if ends[i] - starts[i] < min_duration:
                if i == 0:
                    starts[1] = starts[0]
                    del starts[0], ends[0], labels[0]
                else:
                    ends[i - 1] = ends[i]
                    del starts[i], ends[i], labels[i]
                changed = True
                break
    return starts, ends, labels


def _collapse_adjacent_same_label(starts: list, ends: list, labels: list) -> tuple[list, list, list]:
    """Merging short runs can leave two neighbors with the same label no longer
    separated by anything (A, [short B merged away], A) -- collapse those into one."""
    if not starts:
        return starts, ends, labels
    merged_starts, merged_ends, merged_labels = [starts[0]], [ends[0]], [labels[0]]
    for i in range(1, len(starts)):
        if labels[i] == merged_labels[-1]:
            merged_ends[-1] = ends[i]
        else:
            merged_starts.append(starts[i])
            merged_ends.append(ends[i])
            merged_labels.append(labels[i])
    return merged_starts, merged_ends, merged_labels


def analyze_structure(audio: np.ndarray, sr: int, n_clusters: int = DEFAULT_TIMELINE_CLUSTERS) -> StructureAnalysis:
    """Computes the beat-synced (or framewise-fallback) chroma once, and derives
    both the self-similarity matrix and a clustered timeline from it, so the two
    views are always describing the exact same underlying frames/boundaries."""
    import librosa
    from scipy.sparse import diags
    from sklearn.cluster import AgglomerativeClustering

    duration_sec = len(audio) / sr
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)

    chroma_sync = chroma
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[-1]), sr=sr)
    boundaries = np.append(frame_times, duration_sec)  # n_frames + 1 boundaries -> n_frames intervals

    if len(beat_frames) >= 2:
        synced = librosa.util.sync(chroma, beat_frames)
        if synced.shape[-1] >= MIN_FRAMES_FOR_RECURRENCE:
            chroma_sync = synced
            # librosa.util.sync inserts an implicit leading segment (frame 0 up to
            # the first beat), so synced has len(beat_frames) + 1 columns -- the
            # boundary array needs a matching leading 0.0, not just the beat times.
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            boundaries = np.concatenate([[0.0], beat_times, [duration_sec]])

    # self=False (librosa's default): the main diagonal is deliberately left
    # zeroed. This was flipped to self=True in an earlier version on the theory
    # that a *self*-similarity matrix's diagonal should read as a perfect match --
    # reverted per spec decision: a zeroed diagonal keeps trivial self/near-self
    # similarity from drowning out the real repeated-section stripes, and neither
    # visualization here (timeline or matrix) is designed to rely on the diagonal
    # as a landmark.
    matrix = librosa.segment.recurrence_matrix(chroma_sync, mode="affinity", sym=True).astype(np.float32)

    n_frames = chroma_sync.shape[-1]
    assert len(boundaries) == n_frames + 1, (
        f"boundary/frame mismatch: {len(boundaries)} boundaries for {n_frames} frames"
    )

    k = max(1, min(n_clusters, n_frames))
    if k > 1:
        # tri-diagonal adjacency: frame i only connects to i-1 and i+1, which is
        # what forces every resulting cluster to be one contiguous time interval
        connectivity = diags([1, 1], offsets=[-1, 1], shape=(n_frames, n_frames), dtype=np.int8)
        labels = AgglomerativeClustering(n_clusters=k, connectivity=connectivity, linkage="ward").fit_predict(chroma_sync.T)
    else:
        labels = np.zeros(n_frames, dtype=int)

    seg_starts, seg_ends, seg_labels = [], [], []
    run_start = 0
    for i in range(1, n_frames + 1):
        if i == n_frames or labels[i] != labels[run_start]:
            seg_starts.append(boundaries[run_start])
            seg_ends.append(boundaries[i])
            seg_labels.append(int(labels[run_start]))
            run_start = i

    seg_starts, seg_ends, seg_labels = _merge_short_segments(seg_starts, seg_ends, seg_labels, MIN_SEGMENT_SEC)
    seg_starts, seg_ends, seg_labels = _collapse_adjacent_same_label(seg_starts, seg_ends, seg_labels)

    # Structural confidence: does this song actually repeat in clear sections,
    # or evolve gradually (ambient/through-composed)? The agglomerative
    # segments above always produce *something* (clustering doesn't know how
    # to say "there's no real structure here") -- the novelty curve is what
    # decides whether the UI should trust and show them, or show the honest
    # continuous curve instead. Deliberately kept as a separate signal rather
    # than rebuilding segmentation around novelty-peak boundaries directly:
    # the agglomerative approach is already validated (see module docstring),
    # and confidence-gating it achieves the same "don't force fake sections"
    # goal with far less risk to a working, tuned mechanism.
    novelty_curve = compute_novelty_curve(chroma_sync, duration_sec)
    novelty_times = boundaries[:-1]
    peaks = find_structural_peaks(novelty_curve)
    structural_confidence = compute_structural_confidence(novelty_curve)

    return StructureAnalysis(
        matrix=matrix,
        segment_starts=np.array(seg_starts, dtype=np.float32),
        segment_ends=np.array(seg_ends, dtype=np.float32),
        segment_labels=np.array(seg_labels, dtype=np.int32),
        novelty_curve=novelty_curve,
        novelty_times=novelty_times,
        has_clear_structure=len(peaks) > 0,
        structural_confidence=structural_confidence,
    )


def compute_self_similarity_matrix(audio: np.ndarray, sr: int) -> np.ndarray:
    """Matrix-only convenience wrapper, kept for callers/tests that only need the
    matrix. Prefer analyze_structure() when the timeline is needed too, to avoid
    computing chroma/beat-tracking twice."""
    return analyze_structure(audio, sr).matrix
