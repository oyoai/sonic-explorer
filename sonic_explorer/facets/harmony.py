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

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        return np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)]).astype(np.float32)

    def embed_batch(self, audio_windows: list[np.ndarray], sr: int, batch_size: int = 8) -> np.ndarray:
        """CPU-only chroma extraction -- no real batching benefit (unlike CLAP's
        GPU batching), but keeps the same interface shape run_batch_embedding
        expects across facets."""
        return np.stack([self.embed(window, sr) for window in audio_windows])
