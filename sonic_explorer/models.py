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
