import numpy as np
import pytest

from sonic_explorer.facets.base import Facet
from sonic_explorer.facets.harmony import HarmonyFacet
from sonic_explorer.facets.registry import FacetRegistry, default_registry
from sonic_explorer.facets.sound import SoundFacet


class FakeFacet(Facet):
    """Synthetic facet for testing base similarity()/registry plumbing without any
    heavy deps (torch/librosa) -- those are exercised separately in Colab."""

    name = "fake"
    dim = 4

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0])


def test_similarity_identical_vectors_is_one():
    facet = FakeFacet()
    v = np.array([1.0, 2.0, 3.0])
    assert facet.similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_similarity_orthogonal_vectors_is_zero():
    facet = FakeFacet()
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(facet.similarity(a, b)) < 1e-8


def test_similarity_opposite_vectors_is_minus_one():
    facet = FakeFacet()
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert facet.similarity(a, b) == pytest.approx(-1.0, abs=1e-6)


def test_registry_register_and_get():
    registry = FacetRegistry()
    facet = FakeFacet()
    registry.register(facet)
    assert registry.get("fake") is facet
    assert registry.names() == ["fake"]
    assert registry.all() == [facet]


def test_registry_unknown_name_raises():
    registry = FacetRegistry()
    try:
        registry.get("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_default_registry_has_sound_facet_without_loading_clap():
    # SoundFacet.__init__ must stay lazy -- constructing the registry should never
    # touch torch/transformers, since that's a [colab]-only dependency locally.
    registry = default_registry()
    assert "sound" in registry.names()
    facet = registry.get("sound")
    assert isinstance(facet, SoundFacet)
    assert facet.dim == 512


def test_default_registry_has_harmony_facet():
    registry = default_registry()
    assert "harmony" in registry.names()
    facet = registry.get("harmony")
    assert isinstance(facet, HarmonyFacet)
    assert facet.dim == 24
