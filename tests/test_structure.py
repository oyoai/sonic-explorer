import numpy as np
import pytest

from sonic_explorer.config import CLAP_SR
from sonic_explorer.facets.structure import (
    MIN_SEGMENT_SEC,
    _collapse_adjacent_same_label,
    _merge_short_segments,
    analyze_structure,
    compute_self_similarity_matrix,
)


def make_sine(duration_sec=10.0, freq=440.0, sr=CLAP_SR):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_self_similarity_matrix_is_square_and_symmetric():
    audio = make_sine()
    matrix = compute_self_similarity_matrix(audio, CLAP_SR)

    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert np.allclose(matrix, matrix.T, atol=1e-5)


def test_self_similarity_matrix_has_finite_values():
    audio = make_sine()
    matrix = compute_self_similarity_matrix(audio, CLAP_SR)

    assert np.all(np.isfinite(matrix))
    assert matrix.shape[0] > 1


def test_self_similarity_matrix_diagonal_is_deliberately_zeroed():
    """librosa.segment.recurrence_matrix defaults to self=False, which zeroes the
    main diagonal -- a spec decision, not an oversight: a zeroed diagonal keeps
    trivial self/near-self similarity from drowning out the real repeated-section
    stripes, and neither visualization (timeline or matrix) treats the diagonal as
    a landmark. Pinned as a test so this doesn't silently flip back and forth."""
    audio = make_sine()
    matrix = compute_self_similarity_matrix(audio, CLAP_SR)
    diag = np.diag(matrix)

    assert np.allclose(diag, 0.0)


def test_self_similarity_matrix_falls_back_when_synced_chroma_too_short(monkeypatch):
    """Regression test for a real crash found on real FMA data: some tracks (near-
    silent intros, sparse/ambient sections) produce a beat-synced chroma with too
    few frames for librosa.segment.recurrence_matrix's default width, which raised
    a ParameterError outright. Forces that exact condition deterministically rather
    than hunting for a specific real audio file that triggers it."""
    import librosa

    audio = make_sine(duration_sec=5.0)

    monkeypatch.setattr(librosa.beat, "beat_track", lambda y, sr: (120.0, np.array([0, 1])))
    monkeypatch.setattr(librosa.util, "sync", lambda chroma, beats: chroma[:, :2])  # too-short synced result

    matrix = compute_self_similarity_matrix(audio, CLAP_SR)

    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] > 2  # used the framewise fallback, not the 2-frame synced chroma
    assert np.all(np.isfinite(matrix))


def test_analyze_structure_timeline_covers_full_duration_contiguously():
    audio = make_sine(duration_sec=12.0)
    result = analyze_structure(audio, CLAP_SR)

    assert len(result.segment_starts) == len(result.segment_ends) == len(result.segment_labels)
    assert result.segment_starts[0] == pytest.approx(0.0, abs=1e-3)
    assert result.segment_ends[-1] == pytest.approx(12.0, abs=0.5)
    for i in range(len(result.segment_starts) - 1):
        assert result.segment_ends[i] == pytest.approx(result.segment_starts[i + 1], abs=1e-6)


def test_analyze_structure_timeline_separates_distinct_halves():
    """Sanity check for the boundary/frame-index mapping: two audibly distinct
    halves (a tritone apart -- maximally distant on the chroma circle) should
    produce a label change near the actual midpoint, not some unrelated point,
    which is exactly what an off-by-one in the boundary array would break."""
    sr = CLAP_SR
    half_duration = 6.0
    t = np.linspace(0, half_duration, int(half_duration * sr), endpoint=False)
    first_half = (0.2 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32)  # C4
    second_half = (0.2 * np.sin(2 * np.pi * 369.99 * t)).astype(np.float32)  # F#4
    audio = np.concatenate([first_half, second_half])

    result = analyze_structure(audio, sr, n_clusters=2)

    assert len(set(result.segment_labels.tolist())) == 2
    assert any(abs(start - half_duration) < 1.5 for start in result.segment_starts[1:])


def test_analyze_structure_timeline_segments_meet_minimum_duration():
    """Raw per-beat K-Means labels flip too often to read as clean colored blocks
    (confirmed on real songs: e.g. 22 segments across 27s, most under 1.5s) --
    every emitted segment should meet MIN_SEGMENT_SEC after the merge pass."""
    audio = make_sine(duration_sec=20.0)
    result = analyze_structure(audio, CLAP_SR)
    durations = result.segment_ends - result.segment_starts

    assert durations.min() >= MIN_SEGMENT_SEC - 1e-3


def test_merge_short_segments_absorbs_into_previous():
    """Tests _merge_short_segments in isolation: the short run gets absorbed into
    its previous neighbor (extending that neighbor's end), but merging alone does
    not also collapse same-label neighbors -- that's _collapse_adjacent_same_label's
    job (see the combined-pipeline test below). Using three distinct labels here
    keeps this test isolated to just the absorption behavior."""
    starts = [0.0, 5.0, 5.5, 9.0]
    ends = [5.0, 5.5, 9.0, 12.0]
    labels = [0, 1, 2, 1]  # [5.0, 5.5) is 0.5s -- shorter than min_duration

    m_starts, m_ends, m_labels = _merge_short_segments(starts, ends, labels, min_duration=3.0)

    assert m_starts == [0.0, 5.5, 9.0]
    assert m_ends == [5.5, 9.0, 12.0]
    assert m_labels == [0, 2, 1]  # the short segment (label 1) is gone; its span absorbed into label 0


def test_merge_and_collapse_pipeline_removes_short_middle_segment():
    """The combination actually used by analyze_structure: absorbing a short
    segment can leave two same-label neighbors no longer separated by anything
    (A, [short B absorbed], A) -- the collapse pass then merges those into one."""
    starts = [0.0, 5.0, 5.5, 9.0]
    ends = [5.0, 5.5, 9.0, 12.0]
    labels = [0, 1, 0, 1]  # short [5.0,5.5) sits between two label-0 runs

    m_starts, m_ends, m_labels = _merge_short_segments(starts, ends, labels, min_duration=3.0)
    c_starts, c_ends, c_labels = _collapse_adjacent_same_label(m_starts, m_ends, m_labels)

    assert c_starts == [0.0, 9.0]
    assert c_ends == [9.0, 12.0]
    assert c_labels == [0, 1]


def test_merge_short_segments_absorbs_leading_run_into_next():
    starts = [0.0, 1.0]
    ends = [1.0, 10.0]
    labels = [0, 1]  # first segment is too short and has no previous neighbor

    m_starts, m_ends, m_labels = _merge_short_segments(starts, ends, labels, min_duration=3.0)

    assert m_starts == [0.0]
    assert m_ends == [10.0]
    assert m_labels == [1]


def test_merge_short_segments_leaves_long_segments_alone():
    starts = [0.0, 4.0, 8.0]
    ends = [4.0, 8.0, 12.0]
    labels = [0, 1, 2]

    m_starts, m_ends, m_labels = _merge_short_segments(starts, ends, labels, min_duration=3.0)

    assert m_starts == starts
    assert m_ends == ends
    assert m_labels == labels


def test_collapse_adjacent_same_label_merges_neighbors():
    starts = [0.0, 4.0, 8.0]
    ends = [4.0, 8.0, 12.0]
    labels = [0, 0, 1]

    c_starts, c_ends, c_labels = _collapse_adjacent_same_label(starts, ends, labels)

    assert c_starts == [0.0, 8.0]
    assert c_ends == [8.0, 12.0]
    assert c_labels == [0, 1]


def test_collapse_adjacent_same_label_noop_when_all_distinct():
    starts = [0.0, 4.0, 8.0]
    ends = [4.0, 8.0, 12.0]
    labels = [0, 1, 2]

    c_starts, c_ends, c_labels = _collapse_adjacent_same_label(starts, ends, labels)

    assert c_starts == starts
    assert c_ends == ends
    assert c_labels == labels
