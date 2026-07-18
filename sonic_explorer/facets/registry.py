"""FacetRegistry: one place both the Streamlit UI and (later) agent tools look up
facets by name. Adding a facet later = one new registry entry, not scattered edits."""

from sonic_explorer.facets.base import Facet
from sonic_explorer.facets.sound import SoundFacet


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
    """Core-tier registry: sound facet only. Structure facet joins once its
    Day-3 pipeline lands; harmony joins in Strong tier."""
    registry = FacetRegistry()
    registry.register(SoundFacet())
    return registry
