"""Key and chord estimation from an existing chroma matrix -- both standard,
closed-form techniques (no training, no new model, no new audio processing)
that make analysis.waveform_preview.chroma_for_display's chromagram
genuinely legible instead of just a pretty picture a viewer needs real
music-theory literacy to read.

Plain Python, no Streamlit import -- see spec 8.3's core/interface
separation, same discipline as taste_map.py/network_graph.py."""

from dataclasses import dataclass

import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl & Kessler (1982)'s empirically-derived key profiles -- how
# strongly each scale degree is perceived to belong to a major/minor key,
# relative to a tonic at index 0. The standard, most-widely-reproduced
# values in MIR literature (e.g. music21's KrumhanslSchmuckler analyzer),
# not something invented here.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class KeyEstimate:
    tonic: str  # one of PITCH_CLASSES
    mode: str  # "major" or "minor"
    correlation: float  # Pearson r against the winning profile, [-1, 1] -- how confidently this key fits


def estimate_key(chroma: np.ndarray) -> KeyEstimate:
    """Krumhansl-Schmuckler key-finding: correlate the song's aggregate
    (time-averaged) chroma vector against all 24 major/minor key profiles
    (12 tonics x 2 modes, each the canonical profile rotated to that tonic),
    return the best-correlated match. chroma is (12, T) -- same shape
    chroma_for_display() returns."""
    mean_chroma = chroma.mean(axis=1)

    best_tonic_idx, best_mode, best_r = 0, "major", -2.0
    for tonic_idx in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = np.roll(profile, tonic_idx)
            r = np.corrcoef(mean_chroma, rotated)[0, 1]
            if np.isfinite(r) and r > best_r:
                best_tonic_idx, best_mode, best_r = tonic_idx, mode, r

    return KeyEstimate(tonic=PITCH_CLASSES[best_tonic_idx], mode=best_mode, correlation=float(best_r))


def _chord_templates() -> dict[str, np.ndarray]:
    """24 binary triad templates (12 major + 12 minor root positions) --
    major = root/major-third(+4)/fifth(+7), minor = root/minor-third(+3)/
    fifth(+7), each a 12-element one-hot-per-chord-tone vector."""
    templates = {}
    for root_idx, root in enumerate(PITCH_CLASSES):
        major = np.zeros(12)
        major[[root_idx, (root_idx + 4) % 12, (root_idx + 7) % 12]] = 1.0
        templates[root] = major

        minor = np.zeros(12)
        minor[[root_idx, (root_idx + 3) % 12, (root_idx + 7) % 12]] = 1.0
        templates[f"{root}m"] = minor
    return templates


_CHORD_TEMPLATES = _chord_templates()


def _best_chord_for_frame(frame: np.ndarray) -> str:
    if not np.any(frame):
        return "N/C"  # no discernible chord content in this frame (e.g. silence)
    best_label, best_score = "N/C", -2.0
    for label, template in _CHORD_TEMPLATES.items():
        score = np.corrcoef(frame, template)[0, 1] if np.std(frame) > 0 else 0.0
        if np.isfinite(score) and score > best_score:
            best_label, best_score = label, score
    return best_label


def _mode_filter(labels: list[str], window: int) -> list[str]:
    """Replaces each label with the most common label in a centered window
    around it -- smooths frame-to-frame chord flicker (adjacent near-tied
    frames swapping labels every ~0.1s) into stable runs, the same role a
    median filter plays for continuous signals, applied to categorical
    labels via mode instead. window is forced odd so the center is well
    defined; window=1 is a no-op (returns labels unchanged)."""
    if window <= 1:
        return labels
    if window % 2 == 0:
        window += 1
    half = window // 2
    smoothed = []
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        neighborhood = labels[lo:hi]
        smoothed.append(max(set(neighborhood), key=neighborhood.count))
    return smoothed


@dataclass
class ChordSegment:
    start_sec: float
    end_sec: float
    label: str  # e.g. "C", "Am", or "N/C" for no discernible chord


def estimate_chords(chroma: np.ndarray, times: np.ndarray, smooth_window_sec: float = 1.0) -> list[ChordSegment]:
    """Per-frame chord template matching (24 major/minor triads), smoothed
    with a mode-filter over roughly smooth_window_sec of frames to avoid
    flickery frame-to-frame relabeling, then collapsed into contiguous
    same-chord runs -- a handful of real segments to display, not one label
    per frame. chroma/times are chroma_for_display()'s (12, T) output --
    same data the chromagram already renders, no separate computation."""
    if chroma.shape[1] == 0:
        return []

    raw_labels = [_best_chord_for_frame(chroma[:, i]) for i in range(chroma.shape[1])]

    hop_sec = float(times[1] - times[0]) if len(times) > 1 else smooth_window_sec
    window_frames = max(1, round(smooth_window_sec / hop_sec)) if hop_sec > 0 else 1
    smoothed_labels = _mode_filter(raw_labels, window_frames)

    segments: list[ChordSegment] = []
    seg_start_idx = 0
    for i in range(1, len(smoothed_labels) + 1):
        if i == len(smoothed_labels) or smoothed_labels[i] != smoothed_labels[seg_start_idx]:
            segments.append(ChordSegment(
                start_sec=float(times[seg_start_idx]),
                end_sec=float(times[i]) if i < len(times) else float(times[-1]),
                label=smoothed_labels[seg_start_idx],
            ))
            seg_start_idx = i
    return segments
