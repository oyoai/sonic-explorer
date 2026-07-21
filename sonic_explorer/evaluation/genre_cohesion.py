"""Genre-cohesion evaluation: do a facet's nearest neighbors actually share genre
more often than chance? A defensible, presentable quantitative signal -- not
ground truth for "sounds similar" (see the spec's stated limitation: CLAP captures
a blended notion of sound, and genre is a proxy, not the real target)."""

from dataclasses import dataclass

import numpy as np

from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository
from sonic_explorer.retrieval.song_level_index import build_song_level_index, query_song_level


@dataclass
class GenreCohesionResult:
    facet_name: str
    k: int
    n_queries: int
    observed: float
    random_baseline: float


def genre_cohesion_at_k(
    song_repo: SongRepository,
    embedding_repo: EmbeddingRepository,
    facet_name: str = "sound",
    k: int = 10,
    sample_size: int | None = None,
    seed: int = 42,
) -> GenreCohesionResult:
    rng = np.random.default_rng(seed)
    songs = song_repo.list_songs()
    genre_by_song = {s.id: s.genre_top for s in songs}

    # segment_id -> song_id, restricted to segments actually embedded for this facet
    segment_song: dict[int, int] = {}
    for song in songs:
        for seg in song_repo.get_segments(song.id):
            if embedding_repo.status(seg.id, facet_name) == "done":
                segment_song[seg.id] = song.id

    all_seg_ids = list(segment_song.keys())
    if not all_seg_ids:
        return GenreCohesionResult(facet_name=facet_name, k=k, n_queries=0, observed=0.0, random_baseline=0.0)

    if sample_size is not None and sample_size < len(all_seg_ids):
        query_seg_ids = list(rng.choice(all_seg_ids, size=sample_size, replace=False))
    else:
        query_seg_ids = all_seg_ids

    observed_scores = []
    random_scores = []

    for seg_id in query_seg_ids:
        song_id = segment_song[seg_id]
        query_genre = genre_by_song[song_id]
        query_vec = embedding_repo.get_vector(facet_name, seg_id)

        # observed: real FAISS neighbors, excluding the query's own song
        raw = embedding_repo.search(facet_name, query_vec, k=k + 20)
        neighbors = []
        for cand_id, _ in raw:
            cand_song = segment_song.get(cand_id)
            if cand_song is None or cand_song == song_id:
                continue
            neighbors.append(cand_id)
            if len(neighbors) >= k:
                break
        if neighbors:
            hits = sum(1 for n in neighbors if genre_by_song[segment_song[n]] == query_genre)
            observed_scores.append(hits / len(neighbors))

        # random baseline: k random segments drawn from OTHER songs
        other_seg_ids = [s for s in all_seg_ids if segment_song[s] != song_id]
        if other_seg_ids:
            chosen = rng.choice(other_seg_ids, size=min(k, len(other_seg_ids)), replace=False)
            hits = sum(1 for c in chosen if genre_by_song[segment_song[c]] == query_genre)
            random_scores.append(hits / len(chosen))

    return GenreCohesionResult(
        facet_name=facet_name,
        k=k,
        n_queries=len(query_seg_ids),
        observed=float(np.mean(observed_scores)) if observed_scores else 0.0,
        random_baseline=float(np.mean(random_scores)) if random_scores else 0.0,
    )


def song_level_genre_cohesion_at_k(
    song_repo: SongRepository,
    embedding_repo: EmbeddingRepository,
    facet_name: str = "sound",
    k: int = 10,
    sample_size: int | None = None,
    seed: int = 42,
) -> GenreCohesionResult:
    """Same metric as genre_cohesion_at_k, but querying a song-level index
    (retrieval/song_level_index.py's mean-pooled-per-song vectors) instead of
    individual segments -- the direct comparison retrieval/song_level_index.py's
    aggregation hypothesis needs on the same metric already used to evaluate
    every other facet."""
    from sonic_explorer.analysis.taste_map import mean_pool_song_vectors

    rng = np.random.default_rng(seed)
    songs = song_repo.list_songs()
    genre_by_song = {s.id: s.genre_top for s in songs}

    index = build_song_level_index(song_repo, embedding_repo, facet_name)
    song_vectors = mean_pool_song_vectors(song_repo, embedding_repo, facet_name=facet_name)
    all_song_ids = list(song_vectors.keys())
    if index is None or not all_song_ids:
        return GenreCohesionResult(facet_name=facet_name, k=k, n_queries=0, observed=0.0, random_baseline=0.0)

    if sample_size is not None and sample_size < len(all_song_ids):
        query_song_ids = list(rng.choice(all_song_ids, size=sample_size, replace=False))
    else:
        query_song_ids = all_song_ids

    observed_scores = []
    random_scores = []

    for song_id in query_song_ids:
        song_id = int(song_id)
        query_genre = genre_by_song[song_id]

        results = query_song_level(index, song_vectors[song_id], k=k, exclude_song_id=song_id)
        if results:
            hits = sum(1 for cand_id, _ in results if genre_by_song[cand_id] == query_genre)
            observed_scores.append(hits / len(results))

        other_ids = [sid for sid in all_song_ids if sid != song_id]
        if other_ids:
            chosen = rng.choice(other_ids, size=min(k, len(other_ids)), replace=False)
            hits = sum(1 for c in chosen if genre_by_song[int(c)] == query_genre)
            random_scores.append(hits / len(chosen))

    return GenreCohesionResult(
        facet_name=facet_name,
        k=k,
        n_queries=len(query_song_ids),
        observed=float(np.mean(observed_scores)) if observed_scores else 0.0,
        random_baseline=float(np.mean(random_scores)) if random_scores else 0.0,
    )
