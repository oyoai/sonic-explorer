"""Measures CLAP's sensitivity to pure loudness changes -- a real open
question flagged during a pipeline-normalization audit: no stage in this
app's pipeline normalizes raw audio gain before feature extraction (every
facet's embed() operates on librosa.load()'s output as-is, at whatever gain
the source file happens to be mastered at). Before treating a robustness/
perturbation test's measured similarity changes as reflecting real content
sensitivity, this establishes how much of CLAP's own embedding space moves
under loudness alone, with everything else about the audio held fixed --
so a loudness-driven perturbation doesn't get silently conflated with a
real content change once the fuller perturbation suite is built.

Perturbation here is PURE gain scaling (audio * factor, clipped to [-1, 1]
to avoid digital clipping introducing an actual waveform-shape change at
extreme gain levels) -- no EQ, no compression, no LUFS-style loudness
modeling, since the question is specifically "does a raw gain change move
the embedding," not "does loudness normalization change it" (there is
none in this pipeline to test)."""

from dataclasses import dataclass

import numpy as np

from sonic_explorer.config import CLAP_SR
from sonic_explorer.facets.sound import SoundFacet


@dataclass
class GainDriftResult:
    gain_db: float
    cosine_similarities: list[float]  # one per sampled window: original embedding vs. that window's gain-perturbed embedding

    @property
    def mean_similarity(self) -> float:
        return float(np.mean(self.cosine_similarities)) if self.cosine_similarities else float("nan")

    @property
    def mean_drift(self) -> float:
        """1 - mean cosine similarity -- 0 means the embedding didn't move at all."""
        return 1.0 - self.mean_similarity


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """audio * 10^(gain_db/20), clipped to the valid [-1, 1] sample range.
    Clipping matters at large positive gain_db specifically: without it,
    "loudness perturbation" would silently become "loudness perturbation
    plus hard clipping distortion," a real waveform-shape change that would
    confound the measurement this module exists to isolate."""
    factor = 10.0 ** (gain_db / 20.0)
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


def measure_gain_sensitivity(
    audio_windows: list[np.ndarray],
    gain_levels_db: list[float],
    sr: int = CLAP_SR,
    facet: SoundFacet | None = None,
) -> list[GainDriftResult]:
    """audio_windows: real clips already extracted at sr (this module
    doesn't touch song/segment loading, so it's usable standalone or from
    a script -- see scripts/measure_clap_gain_sensitivity.py for the "pull
    real windows from the library" half). One GainDriftResult per requested
    gain_levels_db entry, each holding one cosine similarity per input
    window -- report the DISTRIBUTION via the result's own list, not just
    mean_drift, since a perturbation's effect can vary a lot clip to clip;
    collapsing to a single number would hide that."""
    facet = facet or SoundFacet()
    baseline_vectors = facet.embed_batch(audio_windows, sr)

    results = []
    for gain_db in gain_levels_db:
        perturbed_windows = [apply_gain_db(w, gain_db) for w in audio_windows]
        perturbed_vectors = facet.embed_batch(perturbed_windows, sr)
        sims = []
        for base_vec, pert_vec in zip(baseline_vectors, perturbed_vectors, strict=True):
            denom = np.linalg.norm(base_vec) * np.linalg.norm(pert_vec) + 1e-9
            sims.append(float(np.dot(base_vec, pert_vec) / denom))
        results.append(GainDriftResult(gain_db=gain_db, cosine_similarities=sims))
    return results
