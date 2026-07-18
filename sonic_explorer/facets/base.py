"""Facet: the Strategy interface every similarity axis (sound, structure, later harmony)
implements. New facet later = one new class + one registry entry, not scattered edits."""

from abc import ABC, abstractmethod

import numpy as np


class Facet(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Embed one audio segment's samples into this facet's vector space."""
        raise NotImplementedError

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity. Shared by all facets -- subclasses only override embed()."""
        a = vec_a / (np.linalg.norm(vec_a) + 1e-8)
        b = vec_b / (np.linalg.norm(vec_b) + 1e-8)
        return float(np.dot(a, b))
