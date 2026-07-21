"""Diagnostic view of retrieval quality beyond genre-cohesion (#7 in the
walkthrough feedback round): does a facet's top-1 match score reliably
separate from noise? A facet whose top-1 scores sit far above random-pair
scores is producing decisive matches; a facet whose top-1 barely beats
random is functionally noise dressed up as a ranked list. Genre-cohesion@k
measures whether neighbors share a label; this measures the raw score
geometry underneath that ranking, which genre-cohesion alone can't reveal
(a facet can beat the random baseline on label-sharing while still having a
nearly flat score landscape, if the shared-label neighbors just happen to
cluster loosely together)."""

from dataclasses import dataclass, field

import numpy as np

from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository
from sonic_explorer.retrieval.song_level_index import build_song_level_index, query_song_level


@dataclass
class ScoreDistribution:
    facet_name: str
    n_queries: int
    top1_scores: list[float]
    random_pair_scores: list[float]
    top1_top2_margins: list[float] = field(default_factory=list)


def top1_score_distribution(
    song_repo: SongRepository,
    embedding_repo: EmbeddingRepository,
    facet_name: str = "sound",
    sample_size: int | None = None,
    seed: int = 42,
) -> ScoreDistribution:
    rng = np.random.default_rng(seed)
    songs = song_repo.list_songs()

    # segment_id -> song_id, restricted to segments actually embedded for this facet
    segment_song: dict[int, int] = {}
    for song in songs:
        for seg in song_repo.get_segments(song.id):
            if embedding_repo.status(seg.id, facet_name) == "done":
                segment_song[seg.id] = song.id

    all_seg_ids = list(segment_song.keys())
    if not all_seg_ids:
        return ScoreDistribution(facet_name=facet_name, n_queries=0, top1_scores=[], random_pair_scores=[])

    if sample_size is not None and sample_size < len(all_seg_ids):
        query_seg_ids = list(rng.choice(all_seg_ids, size=sample_size, replace=False))
    else:
        query_seg_ids = all_seg_ids

    top1_scores: list[float] = []
    random_pair_scores: list[float] = []
    top1_top2_margins: list[float] = []

    for seg_id in query_seg_ids:
        song_id = segment_song[int(seg_id)]
        query_vec = embedding_repo.get_vector(facet_name, int(seg_id))

        # top-2 real neighbors from other songs, excluding the query's own song
        raw = embedding_repo.search(facet_name, query_vec, k=20)
        neighbor_scores = [
            score for cand_id, score in raw if segment_song.get(cand_id) not in (None, song_id)
        ][:2]
        if neighbor_scores:
            top1_scores.append(float(neighbor_scores[0]))
        if len(neighbor_scores) >= 2:
            top1_top2_margins.append(float(neighbor_scores[0] - neighbor_scores[1]))

        # random baseline: one random segment drawn from a different song
        other_seg_ids = [s for s in all_seg_ids if segment_song[s] != song_id]
        if other_seg_ids:
            rand_id = int(rng.choice(other_seg_ids))
            rand_vec = embedding_repo.get_vector(facet_name, rand_id)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(rand_vec) + 1e-9
            random_pair_scores.append(float(np.dot(query_vec, rand_vec) / denom))

    return ScoreDistribution(
        facet_name=facet_name,
        n_queries=len(query_seg_ids),
        top1_scores=top1_scores,
        random_pair_scores=random_pair_scores,
        top1_top2_margins=top1_top2_margins,
    )


def song_level_score_distribution(
    song_repo: SongRepository,
    embedding_repo: EmbeddingRepository,
    facet_name: str = "sound",
    sample_size: int | None = None,
    seed: int = 42,
) -> ScoreDistribution:
    """Same shape/metrics as top1_score_distribution, but querying a
    song-level index (one mean-pooled vector per song, see
    retrieval/song_level_index.py) instead of the segment-level FAISS index
    -- the direct comparison retrieval/song_level_index.py's aggregation
    hypothesis needs: does pooling a song's segments before ranking produce
    a sharper top1-vs-top2 margin than ranking individual segments does?"""
    rng = np.random.default_rng(seed)
    index = build_song_level_index(song_repo, embedding_repo, facet_name)
    if index is None or index.ntotal == 0:
        return ScoreDistribution(facet_name=facet_name, n_queries=0, top1_scores=[], random_pair_scores=[])

    from sonic_explorer.analysis.taste_map import mean_pool_song_vectors

    song_vectors = mean_pool_song_vectors(song_repo, embedding_repo, facet_name=facet_name)
    all_song_ids = list(song_vectors.keys())

    if sample_size is not None and sample_size < len(all_song_ids):
        query_song_ids = list(rng.choice(all_song_ids, size=sample_size, replace=False))
    else:
        query_song_ids = all_song_ids

    top1_scores: list[float] = []
    random_pair_scores: list[float] = []
    top1_top2_margins: list[float] = []

    for song_id in query_song_ids:
        song_id = int(song_id)
        query_vec = song_vectors[song_id]

        results = query_song_level(index, query_vec, k=2, exclude_song_id=song_id)
        if results:
            top1_scores.append(results[0][1])
        if len(results) >= 2:
            top1_top2_margins.append(results[0][1] - results[1][1])

        other_ids = [sid for sid in all_song_ids if sid != song_id]
        if other_ids:
            rand_id = int(rng.choice(other_ids))
            rand_vec = song_vectors[rand_id]
            denom = np.linalg.norm(query_vec) * np.linalg.norm(rand_vec) + 1e-9
            random_pair_scores.append(float(np.dot(query_vec, rand_vec) / denom))

    return ScoreDistribution(
        facet_name=facet_name,
        n_queries=len(query_song_ids),
        top1_scores=top1_scores,
        random_pair_scores=random_pair_scores,
        top1_top2_margins=top1_top2_margins,
    )
