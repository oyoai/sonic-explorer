"""Song and Segment -- the two entities everything else in the package operates on."""

from dataclasses import dataclass, field


@dataclass
class Segment:
    song_id: int
    start_sec: float
    end_sec: float
    segment_index: int
    id: int | None = None


@dataclass
class Song:
    filepath: str
    fma_track_id: int
    title: str
    artist: str
    genre_top: str
    duration_sec: float
    id: int | None = None
    segments: list[Segment] = field(default_factory=list)
    # song DNA (facets/song_dna.py) -- None until the structure batch pipeline
    # computes it; raw (unnormalized) values, see analysis/song_dna.py
    tempo_bpm: float | None = None
    energy: float | None = None
    brightness: float | None = None
    harmonic_complexity: float | None = None
    rhythmic_density: float | None = None
    # bookmarked into "my library" (spec 2.1) -- no separate user/auth model,
    # a single implicit library is enough for this project's scope
    is_saved: bool = False
    # LLM-synthesized natural-language description ("calm piano," "sassy hip
    # hop") from AST tags + DNA -- see pipeline/sound_tagging.py and
    # scripts/generate_song_descriptions.py. None until that batch script runs.
    description: str | None = None
    # Raw AST/AudioSet tags this description was synthesized from, JSON-encoded
    # as [[label, score], ...] (see pipeline/sound_tagging.serialize_tags) --
    # kept separately from `description` because a short synthesized phrase
    # necessarily drops most detected tags, which makes it unreliable for
    # exact sound-content search (e.g. "crow sounds"); the agent's
    # search_by_sound_content tool matches against this field directly.
    sound_tags: str | None = None
