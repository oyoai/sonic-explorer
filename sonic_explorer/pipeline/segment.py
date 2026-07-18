"""Windowing logic -- generalized from the single-song version in
notebooks/audio_deep_dive.ipynb (which used a 1s hop for fine-grained single-song
detail) to fixed library-scale params that bound segment count across ~1,400 tracks."""

from sonic_explorer.config import HOP_SEC, WINDOW_SEC
from sonic_explorer.models import Segment


def segment_song(song_id: int, duration_sec: float, window_sec: float = WINDOW_SEC, hop_sec: float = HOP_SEC) -> list[Segment]:
    """Fixed-size, fixed-hop windows covering [0, duration_sec). Only full-length
    windows are kept -- a trailing partial window is dropped rather than padded,
    to keep every embedded segment the same duration. Songs shorter than one window
    get a single segment covering the whole thing."""
    segments = []
    start = 0.0
    index = 0
    while start + window_sec <= duration_sec:
        segments.append(Segment(song_id=song_id, start_sec=start, end_sec=start + window_sec, segment_index=index))
        start += hop_sec
        index += 1
    if not segments:
        segments.append(Segment(song_id=song_id, start_sec=0.0, end_sec=duration_sec, segment_index=0))
    return segments
