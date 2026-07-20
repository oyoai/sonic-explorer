"""Stem-separated facets (spec section 3, Stretch tier): vocal/drums/bass/
instrumental similarity. Each is exactly SoundFacet run on an isolated audio
stream instead of the full mix -- the same CLAP embedding logic, no new
architecture, only a different registered name. The actual separation
(Demucs) happens upstream in pipeline/separation.py during the batch
pipeline, not here -- these classes never see the full mix, only whatever
audio they're handed."""

from sonic_explorer.facets.sound import SoundFacet


class VocalFacet(SoundFacet):
    name = "vocal"


class DrumsFacet(SoundFacet):
    name = "drums"


class BassFacet(SoundFacet):
    name = "bass"


class InstrumentalFacet(SoundFacet):
    name = "instrumental"
