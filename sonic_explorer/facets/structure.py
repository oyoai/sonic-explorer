"""Structure: a song's self-similarity matrix (verse/chorus shape) and a compact
segmented timeline (which stretches of the song sound alike) -- promoted from the
beat-synced chroma recurrence matrix (audio_deep_dive.ipynb cell 21) and the
clustering-into-a-template idea (cell 16, DBSCAN there; K-Means here for
predictable behavior across 1400 songs without per-song eps tuning).

Powers Song X-Ray: the timeline is the primary, non-technical view (a color bar
where matching colors = similar-sounding sections); the matrix is the secondary
"technical detail" view underneath. Core's UI does not do structure-based
retrieval (see the spec's Core/Strong split), so none of this routes through the
Facet/FacetRegistry retrieval path the way SoundFacet does -- these are song-level
artifacts computed and cached directly.
"""

from dataclasses import dataclass

import numpy as np

# librosa.segment.recurrence_matrix's default width needs (n_frames - 1) // 2 >= 1,
# i.e. at least 3 frames -- a real failure mode on real FMA data: some tracks (near-
# silent intros, sparse/ambient sections, non-rhythmic clips) produce a beat-synced
# chroma with only 1-2 frames, which crashes recurrence_matrix outright.
MIN_FRAMES_FOR_RECURRENCE = 3

DEFAULT_TIMELINE_CLUSTERS = 6

# Per-beat K-Means labels flip frequently even within one perceptual "section" --
# raw output averages well under a second per run (confirmed on real songs: e.g.
# 22 segments across 27s, most under 1.5s). Unreadable as a colored block view for
# a non-technical viewer, who needs phrase-scale chunks, not per-beat confetti.
MIN_SEGMENT_SEC = 3.0


@dataclass
class StructureTimeline:
    """Just the segmented-timeline half of StructureAnalysis -- what
    EmbeddingRepository.get_structure_timeline() reads back from disk (the
    matrix and timeline are persisted as separate files, see
    pipeline/build_structure_library.py)."""

    segment_starts: np.ndarray
    segment_ends: np.ndarray
    segment_labels: np.ndarray


@dataclass
class StructureAnalysis:
    matrix: np.ndarray  # self-similarity matrix, frames x frames (diagonal == max)
    segment_starts: np.ndarray  # start_sec per contiguous timeline segment
    segment_ends: np.ndarray  # end_sec per contiguous timeline segment
    segment_labels: np.ndarray  # cluster label (int) per contiguous timeline segment


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
    from sklearn.cluster import KMeans

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

    # self=True: librosa's default (False) explicitly zeroes the main diagonal --
    # correct for recurrence/repeat-finding, wrong for a *self*-similarity matrix.
    matrix = librosa.segment.recurrence_matrix(chroma_sync, mode="affinity", sym=True, self=True).astype(np.float32)

    n_frames = chroma_sync.shape[-1]
    assert len(boundaries) == n_frames + 1, (
        f"boundary/frame mismatch: {len(boundaries)} boundaries for {n_frames} frames"
    )

    k = max(1, min(n_clusters, n_frames))
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(chroma_sync.T) if k > 1 else np.zeros(n_frames, dtype=int)

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

    return StructureAnalysis(
        matrix=matrix,
        segment_starts=np.array(seg_starts, dtype=np.float32),
        segment_ends=np.array(seg_ends, dtype=np.float32),
        segment_labels=np.array(seg_labels, dtype=np.int32),
    )


def compute_self_similarity_matrix(audio: np.ndarray, sr: int) -> np.ndarray:
    """Matrix-only convenience wrapper, kept for callers/tests that only need the
    matrix. Prefer analyze_structure() when the timeline is needed too, to avoid
    computing chroma/beat-tracking twice."""
    return analyze_structure(audio, sr).matrix
