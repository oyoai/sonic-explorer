"""Corpus-wide z-score whitening for facet embeddings -- a post-hoc fix for
harmony's collapsed embedding space. retrieval_diagnostics.py's score
distributions found harmony's random-pair baseline sits at cosine similarity
0.85-0.95: raw chroma-derived vectors have very little natural spread
across the corpus, so real differences barely register once L2-normalized
for cosine search. Whitening each dimension to zero mean / unit variance
before re-normalizing spreads the corpus out along directions that actually
vary, without re-extracting any audio features -- this operates purely on
vectors already sitting in a FAISS index.

Generic over any facet's embedding, not harmony-specific -- fit once on a
corpus of same-facet vectors, then transform each vector independently."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Whitener:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, vector: np.ndarray) -> np.ndarray:
        whitened = (vector - self.mean) / self.std
        norm = np.linalg.norm(whitened)
        if norm > 0:
            whitened = whitened / norm
        return whitened.astype(np.float32)


def fit_whitener(vectors: list[np.ndarray], eps: float = 1e-6) -> Whitener:
    """vectors should be the corpus's current (already L2-normalized, as
    stored in the FAISS index) embeddings -- fitting mean/std on those is
    exactly what needs undoing, not the raw pre-normalization features,
    since normalization is what collapsed the natural spread in the first
    place for a low-dimensional, mostly-positive space like chroma."""
    matrix = np.stack(vectors)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < eps, 1.0, std)  # a near-constant dimension carries no signal -- leave it unscaled, not blown up
    return Whitener(mean=mean, std=std)
