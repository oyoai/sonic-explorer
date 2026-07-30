"""Harmony facet: chroma features -- key, chord color, tonal similarity. A
second, independently-computed matching axis alongside SoundFacet (spec 3's
"several kinds of similarity" -- general-purpose CLAP embeddings blend sound
qualities together and don't cleanly separate out harmony, so a genuinely
harmony-only comparison needs its own targeted feature).

Unlike structure.py's artifacts, this routes through the real Facet/
FacetRegistry/RetrievalService retrieval path the way SoundFacet does --
harmony similarity is meant to be queried and toggled to in Moment Matcher,
not just displayed as a per-song visualization."""

import numpy as np

from sonic_explorer.facets.base import Facet

CHROMA_BINS = 12
HARMONY_DIM = CHROMA_BINS * 2  # mean + std per pitch class


class HarmonyFacet(Facet):
    name = "harmony"
    dim = HARMONY_DIM

    def __init__(self):
        # See analysis/embedding_whitening.py's module docstring for why
        # this is loaded here rather than left as a one-off script side
        # effect: every harmony vector actually sitting in the FAISS index
        # went through this same whitening transform, so a fresh embed()
        # call (a live query, a new song, a robustness test) needs to apply
        # the identical transform to land in the same space -- not just an
        # unwhitened raw vector that happens to also be L2-normalized.
        self._whitener = self._load_whitener()

    @staticmethod
    def _load_whitener():
        from sonic_explorer.analysis.embedding_whitening import Whitener
        from sonic_explorer.config import HARMONY_WHITENER_PATH

        if not HARMONY_WHITENER_PATH.exists():
            return None
        return Whitener.load(HARMONY_WHITENER_PATH)

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        vector = np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)]).astype(np.float32)
        if self._whitener is None:
            return vector
        # fit_whitener() was fit on the corpus's ALREADY L2-normalized
        # vectors (see its own docstring) -- replicating that exact order
        # here (normalize, then whiten) is what makes this match the
        # index's stored vectors bit-for-bit, not just approximately.
        # Whitener.transform() re-normalizes internally afterward, so the
        # final output is unit-norm either way.
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return self._whitener.transform(vector)

    def embed_batch(self, audio_windows: list[np.ndarray], sr: int, batch_size: int = 8) -> np.ndarray:
        """CPU-only chroma extraction -- no real batching benefit (unlike CLAP's
        GPU batching), but keeps the same interface shape run_batch_embedding
        expects across facets."""
        return np.stack([self.embed(window, sr) for window in audio_windows])
