import numpy as np
import pytest

from sonic_explorer.facets.base import Facet
from sonic_explorer.facets.harmony import HarmonyFacet
from sonic_explorer.facets.registry import FacetRegistry, default_registry
from sonic_explorer.facets.sound import SoundFacet
from sonic_explorer.facets.stems import BassFacet, DrumsFacet, InstrumentalFacet, VocalFacet
from sonic_explorer.facets.tags import SoundTagsFacet, tags_to_text


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
        raise AssertionError("expected KeyError")
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


def test_default_registry_has_stem_facets_without_loading_clap():
    registry = default_registry()
    expected = {"vocal": VocalFacet, "drums": DrumsFacet, "bass": BassFacet, "instrumental": InstrumentalFacet}
    for name, cls in expected.items():
        assert name in registry.names()
        facet = registry.get(name)
        assert isinstance(facet, cls)
        assert facet.dim == 512  # inherits SoundFacet's CLAP dim -- same embedding logic, isolated audio


def test_default_registry_has_sound_tags_facet_without_loading_clap():
    # Constructing SoundTagsFacet must stay lazy too -- same discipline as sound/stems.
    registry = default_registry()
    assert "sound_tags" in registry.names()
    facet = registry.get("sound_tags")
    assert isinstance(facet, SoundTagsFacet)
    assert facet.dim == 512  # CLAP's joint text-audio space, same as the Sound facet


def test_tags_to_text_joins_labels_highest_confidence_first():
    tags = [("Cello", 0.259), ("Bowed string instrument", 0.155), ("Violin, fiddle", 0.096)]
    assert tags_to_text(tags) == "Cello, Bowed string instrument, Violin, fiddle"


def test_tags_to_text_handles_empty_list():
    assert tags_to_text([]) == ""


def test_tags_to_text_excludes_generic_umbrella_labels():
    tags = [("Music", 0.35), ("Cello", 0.15), ("Musical instrument", 0.09), ("Violin, fiddle", 0.07)]
    assert tags_to_text(tags) == "Cello, Violin, fiddle"


def test_tags_to_text_all_generic_returns_empty_string():
    assert tags_to_text([("Music", 0.5), ("Musical instrument", 0.1)]) == ""
