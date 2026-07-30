import numpy as np

from sonic_explorer.analysis.key_chord import (
    _MAJOR_PROFILE,
    _MINOR_PROFILE,
    ChordSegment,
    estimate_chords,
    estimate_key,
)


def test_estimate_key_identifies_c_major_from_the_c_major_profile_itself():
    """The cleanest possible case: a mean chroma vector literally
    proportional to the canonical C-major Krumhansl-Kessler profile must be
    identified as C major, with a strong positive correlation."""
    chroma = np.tile(_MAJOR_PROFILE.reshape(12, 1), (1, 5))

    result = estimate_key(chroma)

    assert result.tonic == "C"
    assert result.mode == "major"
    assert result.correlation > 0.99


def test_estimate_key_identifies_a_rotated_tonic():
    """Rotating the major profile by 7 semitones (a perfect fifth) simulates
    a song centered on G rather than C -- the rotation itself, not a new
    profile, is what should drive the tonic identification."""
    rotated = np.roll(_MAJOR_PROFILE, 7)
    chroma = np.tile(rotated.reshape(12, 1), (1, 5))

    result = estimate_key(chroma)

    assert result.tonic == "G"
    assert result.mode == "major"


def test_estimate_key_distinguishes_major_from_minor():
    chroma = np.tile(_MINOR_PROFILE.reshape(12, 1), (1, 5))

    result = estimate_key(chroma)

    assert result.tonic == "C"
    assert result.mode == "minor"


def test_estimate_chords_labels_a_pure_c_major_triad_frame():
    frame = np.zeros(12)
    frame[[0, 4, 7]] = 1.0  # C, E, G
    chroma = np.tile(frame.reshape(12, 1), (1, 3))
    times = np.array([0.0, 0.5, 1.0])

    segments = estimate_chords(chroma, times, smooth_window_sec=0.1)

    assert len(segments) == 1
    assert segments[0].label == "C"


def test_estimate_chords_labels_a_minor_triad_frame():
    frame = np.zeros(12)
    frame[[9, 0, 4]] = 1.0  # A, C, E -- A minor
    chroma = np.tile(frame.reshape(12, 1), (1, 3))
    times = np.array([0.0, 0.5, 1.0])

    segments = estimate_chords(chroma, times, smooth_window_sec=0.1)

    assert segments[0].label == "Am"


def test_estimate_chords_smoothing_removes_a_single_flickering_frame():
    """Mostly C major with one lone outlier frame in the middle -- the
    mode-filter smoothing must absorb that flicker into the surrounding
    majority rather than reporting three separate tiny segments."""
    c_major = np.zeros(12)
    c_major[[0, 4, 7]] = 1.0
    g_major = np.zeros(12)
    g_major[[7, 11, 2]] = 1.0

    frames = [c_major] * 4 + [g_major] + [c_major] * 4
    chroma = np.stack(frames, axis=1)
    times = np.arange(len(frames), dtype=float) * 0.1  # 0.1s hops

    segments = estimate_chords(chroma, times, smooth_window_sec=1.0)  # window spans the whole clip

    assert len(segments) == 1
    assert segments[0].label == "C"


def test_estimate_chords_without_smoothing_keeps_the_flicker():
    """Sanity check that the smoothing test above is actually testing
    something -- a near-zero smoothing window should NOT absorb the outlier
    frame, confirming the smoothed result differs because of the window,
    not because chord detection itself ignores that frame."""
    c_major = np.zeros(12)
    c_major[[0, 4, 7]] = 1.0
    g_major = np.zeros(12)
    g_major[[7, 11, 2]] = 1.0

    frames = [c_major] * 4 + [g_major] + [c_major] * 4
    chroma = np.stack(frames, axis=1)
    times = np.arange(len(frames), dtype=float) * 0.1

    segments = estimate_chords(chroma, times, smooth_window_sec=0.05)  # sub-one-frame window -- effectively no smoothing

    assert len(segments) == 3  # C run, G outlier, C run


def test_estimate_chords_segments_cover_contiguous_time_ranges():
    c_major = np.zeros(12)
    c_major[[0, 4, 7]] = 1.0
    frames = [c_major] * 4
    chroma = np.stack(frames, axis=1)
    times = np.array([0.0, 0.5, 1.0, 1.5])

    segments = estimate_chords(chroma, times, smooth_window_sec=0.1)

    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 1.5


def test_estimate_chords_handles_empty_chroma():
    assert estimate_chords(np.zeros((12, 0)), np.zeros(0)) == []


def test_estimate_chords_silent_frame_is_no_chord():
    chroma = np.zeros((12, 3))
    times = np.array([0.0, 0.5, 1.0])

    segments = estimate_chords(chroma, times, smooth_window_sec=0.1)

    assert segments[0].label == "N/C"


def test_chord_segment_is_a_plain_dataclass():
    seg = ChordSegment(start_sec=0.0, end_sec=1.0, label="C")
    assert seg.start_sec == 0.0
    assert seg.label == "C"
