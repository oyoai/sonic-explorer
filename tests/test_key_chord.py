import numpy as np
import pytest

from sonic_explorer.analysis.key_chord import (
    MIN_SEGMENT_SEC_DEFAULT,
    MIN_SEGMENT_SEC_FLOOR,
    _MAJOR_PROFILE,
    _MINOR_PROFILE,
    ChordSegment,
    _tempo_relative_min_segment_sec,
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
    not because chord detection itself ignores that frame. min_segment_sec
    is disabled here too, since this test targets the mode filter
    specifically -- the merge pass is covered separately below."""
    c_major = np.zeros(12)
    c_major[[0, 4, 7]] = 1.0
    g_major = np.zeros(12)
    g_major[[7, 11, 2]] = 1.0

    frames = [c_major] * 4 + [g_major] + [c_major] * 4
    chroma = np.stack(frames, axis=1)
    times = np.arange(len(frames), dtype=float) * 0.1

    segments = estimate_chords(chroma, times, smooth_window_sec=0.05, min_segment_sec=0)  # no smoothing, no merge

    assert len(segments) == 3  # C run, G outlier, C run


def test_estimate_chords_merges_a_short_segment_into_its_longer_neighbor():
    """The residual-flicker case the mode filter alone can't clean up: a
    short-lived segment (below min_segment_sec) sitting between two longer
    runs of different chords gets absorbed into whichever neighbor is
    longer, rather than surviving as its own tiny segment."""
    c_major = np.zeros(12)
    c_major[[0, 4, 7]] = 1.0
    g_major = np.zeros(12)
    g_major[[7, 11, 2]] = 1.0

    # No smoothing (isolate the merge pass): C run, one-frame G blip, C run
    # -- the blip is shorter than min_segment_sec and gets absorbed into
    # whichever C run is longer (the first one, 4 frames vs 4 frames minus
    # the trailing partial -- see the exact-duration assertions below).
    frames = [c_major] * 4 + [g_major] + [c_major] * 4
    chroma = np.stack(frames, axis=1)
    times = np.arange(len(frames), dtype=float) * 0.1  # 0.1s hops; G blip alone is 0.1s wide

    segments = estimate_chords(chroma, times, smooth_window_sec=0.05, min_segment_sec=0.15)

    assert len(segments) == 1
    assert segments[0].label == "C"
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == times[-1]


def test_estimate_chords_merge_pass_prefers_the_longer_neighbor():
    """When a short segment sits between two unequal-length neighbors, it
    should be absorbed into the longer one, not just whichever comes
    first -- confirms the merge pass actually compares durations rather
    than always merging left (or always merging right)."""
    c_major = np.zeros(12)
    c_major[[0, 4, 7]] = 1.0
    g_major = np.zeros(12)
    g_major[[7, 11, 2]] = 1.0
    a_minor = np.zeros(12)
    a_minor[[9, 0, 4]] = 1.0

    # Short C run (2 frames), G blip (1 frame), long A-minor run (6 frames).
    # The blip is shorter than min_segment_sec and should merge into the
    # longer A-minor neighbor, not the shorter C run.
    frames = [c_major] * 2 + [g_major] * 1 + [a_minor] * 6
    chroma = np.stack(frames, axis=1)
    times = np.arange(len(frames), dtype=float) * 0.1

    segments = estimate_chords(chroma, times, smooth_window_sec=0.05, min_segment_sec=0.15)

    assert len(segments) == 2
    assert segments[0].label == "C"
    assert segments[1].label == "Am"
    assert segments[1].start_sec == 0.2  # G blip's span absorbed into the A-minor segment


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


def test_tempo_relative_min_segment_sec_scales_with_beat_duration():
    """A 120 BPM song has a 0.5s beat -- one beat is the minimum plausible
    chord duration (MIN_SEGMENT_BEATS=1), and 0.5s is well above the
    absolute floor, so it should pass through unclamped."""
    assert _tempo_relative_min_segment_sec(120.0) == pytest.approx(0.5)


def test_tempo_relative_min_segment_sec_clamps_to_the_absolute_floor():
    """A very fast song's beat is shorter than MIN_SEGMENT_SEC_FLOOR --
    the floor wins rather than letting the tempo-relative number go
    arbitrarily low."""
    fast_bpm = 60.0 / (MIN_SEGMENT_SEC_FLOOR / 2)  # beat duration is half the floor
    assert _tempo_relative_min_segment_sec(fast_bpm) == pytest.approx(MIN_SEGMENT_SEC_FLOOR)


@pytest.mark.parametrize("bad_tempo", [None, 0.0, -10.0])
def test_tempo_relative_min_segment_sec_falls_back_without_a_real_tempo(bad_tempo):
    assert _tempo_relative_min_segment_sec(bad_tempo) == MIN_SEGMENT_SEC_DEFAULT


def test_estimate_chords_derives_min_segment_sec_from_tempo_bpm_when_not_given_explicitly():
    """A short G blip between two longer C runs: at a slow tempo (60 BPM,
    1s beat) the blip is shorter than the tempo-relative floor and gets
    merged away, exactly like passing an equivalent min_segment_sec by
    hand would -- confirms tempo_bpm actually drives the merge pass, not
    just accepted and ignored."""
    c_major = np.zeros(12)
    c_major[[0, 4, 7]] = 1.0
    g_major = np.zeros(12)
    g_major[[7, 11, 2]] = 1.0

    frames = [c_major] * 4 + [g_major] + [c_major] * 4
    chroma = np.stack(frames, axis=1)
    times = np.arange(len(frames), dtype=float) * 0.1  # G blip alone is 0.1s wide

    segments = estimate_chords(chroma, times, smooth_window_sec=0.05, tempo_bpm=60.0)

    assert len(segments) == 1
    assert segments[0].label == "C"
