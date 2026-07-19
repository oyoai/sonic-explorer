"""Corpus-wide normalization for song DNA (facets/song_dna.py's raw stats) --
maps each song's raw scalars onto a comparable [0,1] scale for the radar chart,
since e.g. "brightness" in raw Hz is meaningless without knowing the library's
actual range. Plain Python, no Streamlit import (core/interface separation,
spec 8.3), mirroring analysis/taste_map.py's pattern."""

from dataclasses import dataclass

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
