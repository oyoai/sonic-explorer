"""Linear probing: how much of each Song DNA scalar (tempo, energy,
brightness, harmonic complexity, rhythmic density) is linearly recoverable
from CLAP's general-purpose sound embedding alone? A real probe, not a
tautology -- CLAP was never trained to predict any of these specific
scalars (they're independently computed librosa features, see
facets/song_dna.py), so a probe finding real linear structure is genuine
evidence CLAP's embedding space geometrically encodes that property, not
just some unrelated correlate of it. Lighter-weight supporting evidence
than genre_free_clustering.py: it says something about what CLAP's
embedding encodes, not directly about the genre-vs-audio thesis the
clustering result speaks to.

Ridge regression, not plain OLS: with only a few hundred songs and a
512-dim embedding, ordinary least squares is badly underdetermined
(far more features than samples) and would overfit to a meaningless,
unstable "perfect" in-sample fit. Ridge's L2 penalty keeps the fit
stable; K-fold CROSS-VALIDATED R^2 (never fit-and-score on the same
data) is what's actually reported here, so a high number reflects real
out-of-sample predictive power, not memorization of the training set."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

DEFAULT_CV_FOLDS = 5
DEFAULT_ALPHA = 10.0


@dataclass
class LinearProbeResult:
    axis: str
    n_songs: int
    r2_mean: float
    r2_std: float
    fold_scores: list[float]


def probe_dna_from_embeddings(
    song_vectors: dict[int, np.ndarray],
    dna_by_song: dict[int, dict[str, float]],
    axes: list[str],
    cv_folds: int = DEFAULT_CV_FOLDS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> dict[str, LinearProbeResult]:
    """dna_by_song: song_id -> {"tempo_bpm": ..., "energy": ..., ...},
    RAW (unnormalized) DNA scalars -- Ridge doesn't need pre-normalized
    targets, and R^2 is scale-invariant either way. A song missing a given
    axis's value is excluded from THAT axis's probe only, not from every
    axis (different axes can have different valid song sets)."""
    results = {}
    for axis in axes:
        song_ids = [
            sid for sid in song_vectors
            if sid in dna_by_song and dna_by_song[sid].get(axis) is not None
        ]
        matrix = np.stack([song_vectors[sid] for sid in song_ids])
        target = np.array([dna_by_song[sid][axis] for sid in song_ids])

        n_splits = min(cv_folds, len(song_ids))
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(Ridge(alpha=alpha), matrix, target, cv=kfold, scoring="r2")

        results[axis] = LinearProbeResult(
            axis=axis, n_songs=len(song_ids),
            r2_mean=float(np.mean(scores)), r2_std=float(np.std(scores)),
            fold_scores=[float(s) for s in scores],
        )
    return results
