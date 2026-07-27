"""FacetRegistry: one place both the Streamlit UI and (later) agent tools look up
facets by name. Adding a facet later = one new registry entry, not scattered edits."""

from sonic_explorer.facets.base import Facet
from sonic_explorer.facets.harmony import HarmonyFacet
from sonic_explorer.facets.sound import SoundFacet
from sonic_explorer.facets.stems import BassFacet, DrumsFacet, InstrumentalFacet, VocalFacet
from sonic_explorer.facets.tags import SoundTagsFacet


class FacetRegistry:
    def __init__(self):
        self._facets: dict[str, Facet] = {}

    def register(self, facet: Facet) -> None:
        self._facets[facet.name] = facet

    def get(self, name: str) -> Facet:
        if name not in self._facets:
            raise KeyError(f"No facet registered under name {name!r}. Registered: {list(self._facets)}")
        return self._facets[name]

    def all(self) -> list[Facet]:
        return list(self._facets.values())

    def names(self) -> list[str]:
        return list(self._facets.keys())


def default_registry() -> FacetRegistry:
    """Sound (Core), harmony (Strong), the four stem-separated facets
    (Stretch -- vocal/drums/bass/instrumental, see facets/stems.py), and
    sound_tags (per-segment AST-detected tags, embedded via CLAP's text
    encoder -- see facets/tags.py) all route through the real retrieval
    path. Registering a facet here before its data actually exists is
    deliberate: the UI (Explore's facet toggle) reads options from this
    registry, not a hardcoded list, so a new facet shows up immediately and
    just reports "nothing embedded yet" until its batch pipeline actually
    runs -- same pattern every facet here has followed during its own
    build, most recently sound_tags (notebooks/11_sound_tags_facet.ipynb).
    Structure is deliberately not registered here: its artifacts
    (self-similarity matrix, timeline, fingerprints) are song-level
    visualizations, not per-segment retrieval vectors -- see
    facets/structure.py."""
    registry = FacetRegistry()
    registry.register(SoundFacet())
    registry.register(HarmonyFacet())
    registry.register(VocalFacet())
    registry.register(DrumsFacet())
    registry.register(BassFacet())
    registry.register(InstrumentalFacet())
    registry.register(SoundTagsFacet())
    return registry
