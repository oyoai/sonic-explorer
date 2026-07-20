"""Corpus-wide normalization for song DNA (facets/song_dna.py's raw stats) --
maps each song's raw scalars onto a comparable [0,1] scale for the radar chart,
since e.g. "brightness" in raw Hz is meaningless without knowing the library's
actual range. Plain Python, no Streamlit import (core/interface separation,
spec 8.3), mirroring analysis/taste_map.py's pattern.

nearest_songs_by_dna() powers "radar chart as query" (spec 2.3): a user-drawn
target profile lives in this exact same normalized [0,1] axis space every
song already gets mapped into for the static overlay, so finding nearby songs
is just nearest-neighbor search over that space -- no new infrastructure."""

from dataclasses import dataclass

import numpy as np

AXES = ["tempo_bpm", "energy", "brightness", "harmonic_complexity", "rhythmic_density"]
AXIS_LABELS = {
    "tempo_bpm": "Tempo",
    "energy": "Energy",
    "brightness": "Brightness",
    "harmonic_complexity": "Harmonic Complexity",
    "rhythmic_density": "Rhythmic Density",
}


@dataclass
class DNANormalizer:
    """Min-max bounds per axis, fit once across every song in the corpus that
    has song DNA computed. Simpler than percentile-rank and sufficient here --
    the radar chart only needs "where does this song sit within the library's
    actual range," not outlier-robust statistics."""

    mins: dict[str, float]
    maxs: dict[str, float]

    def normalize(self, raw: dict[str, float | None]) -> dict[str, float]:
        out = {}
        for axis in AXES:
            lo, hi = self.mins[axis], self.maxs[axis]
            val = raw.get(axis)
            out[axis] = (val - lo) / (hi - lo) if val is not None and hi > lo else 0.0
        return out


def fit_normalizer(all_raw_stats: list[dict[str, float | None]]) -> DNANormalizer:
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    for axis in AXES:
        values = [s[axis] for s in all_raw_stats if s.get(axis) is not None]
        # mins == maxs (both 0.0) when there's no data to fit a range from --
        # DNANormalizer.normalize()'s hi > lo check then correctly falls back to
        # 0.0 for every song, instead of an arbitrary 0..1 range that would
        # produce misleadingly large normalized values for any raw value > 1.
        mins[axis] = min(values) if values else 0.0
        maxs[axis] = max(values) if values else 0.0
    return DNANormalizer(mins=mins, maxs=maxs)


@dataclass
class DNAMatch:
    song_id: int
    distance: float  # euclidean distance in normalized [0,1]^5 axis space -- lower is closer


def nearest_songs_by_dna(
    target: dict[str, float], normalized_by_song: dict[int, dict[str, float]], k: int = 6
) -> list[DNAMatch]:
    """target and every value in normalized_by_song must already be normalized
    (DNANormalizer.normalize() output) -- this does no normalization itself,
    just distance + ranking, so it stays usable for both a real song's DNA and
    a user-hand-drawn target with no notion of "raw" units."""
    target_vec = np.array([target[axis] for axis in AXES])
    scored = [
        DNAMatch(song_id=song_id, distance=float(np.linalg.norm(np.array([norm[axis] for axis in AXES]) - target_vec)))
        for song_id, norm in normalized_by_song.items()
    ]
    scored.sort(key=lambda m: m.distance)
    return scored[:k]
