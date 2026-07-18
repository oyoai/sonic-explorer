"""The one source of truth both the Streamlit UI and (later) the agent's tools
query through -- nothing outside this file calls EmbeddingRepository.search directly."""

from dataclasses import dataclass

import numpy as np

from sonic_explorer.models import Segment, Song
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository


@dataclass
class Match:
    segment: Segment
    song: Song
    score: float


class RetrievalService:
    def __init__(self, song_repo: SongRepository, embedding_repo: EmbeddingRepository):
        self.song_repo = song_repo
        self.embedding_repo = embedding_repo

    def query_by_segment(
        self, segment_id: int, facet_name: str = "sound", k: int = 10, exclude_same_song: bool = True
    ) -> list[Match]:
        """Snaps to a precomputed segment's vector rather than re-embedding on the
        fly -- Core has no local CLAP inference path, only the synced FAISS index."""
        vector = self.embedding_repo.get_vector(facet_name, segment_id)
        exclude_song_id = None
        if exclude_same_song:
            exclude_song_id = self.song_repo.get_segment(segment_id).song_id
        return self.query_by_vector(
            vector, facet_name=facet_name, k=k, exclude_segment_id=segment_id, exclude_song_id=exclude_song_id
        )

    def query_by_vector(
        self,
        vector: np.ndarray,
        facet_name: str = "sound",
        k: int = 10,
        exclude_segment_id: int | None = None,
        exclude_song_id: int | None = None,
    ) -> list[Match]:
        # over-fetch so filtering out the query segment/song doesn't leave us short of k
        raw = self.embedding_repo.search(facet_name, vector, k=k + 10)
        matches: list[Match] = []
        for seg_id, score in raw:
            if seg_id == exclude_segment_id:
                continue
            segment = self.song_repo.get_segment(seg_id)
            if exclude_song_id is not None and segment.song_id == exclude_song_id:
                continue
            song = self.song_repo.get_song(segment.song_id)
            matches.append(Match(segment=segment, song=song, score=score))
            if len(matches) >= k:
                break
        return matches
