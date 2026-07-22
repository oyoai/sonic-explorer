"""Generates a diverse set of segment pairs for human similarity rating --
the calibration_ratings dataset section 9's blend-weight regression and
(conditionally) CLAP fine-tuning depend on. Deliberately not uniform random:
a naive random sample over ~14,600 segments skews almost entirely to
"obviously dissimilar" (see evaluation/retrieval_diagnostics.py's
random-pair-baseline numbers -- most facets sit well below 0.5 cosine for a
random pair), giving a rating set with little real variance to learn from.
Instead draws from three bands off the sound facet's real retrieval results
-- high similarity (real top-1 matches), medium (rank ~10), and random --
so the resulting ratings actually span the similarity range a regression or
fine-tuning objective needs to learn anything from."""

from dataclasses import dataclass

import numpy as np

from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository


@dataclass
class CalibrationPair:
    segment_a_id: int
    segment_b_id: int
    band: str  # "high" | "medium" | "random" -- for reporting spread, never shown to the rater


def generate_calibration_pairs(
    song_repo: SongRepository,
    embedding_repo: EmbeddingRepository,
    facet_name: str = "sound",
    n_high: int = 120,
    n_medium: int = 110,
    n_random: int = 120,
    seed: int = 42,
) -> list[CalibrationPair]:
    rng = np.random.default_rng(seed)
    songs = song_repo.list_songs()

    segment_song: dict[int, int] = {}
    for song in songs:
        for seg in song_repo.get_segments(song.id):
            if embedding_repo.status(seg.id, facet_name) == "done":
                segment_song[seg.id] = song.id

    all_seg_ids = list(segment_song.keys())
    if not all_seg_ids:
        return []

    pairs: list[CalibrationPair] = []
    seen_pairs: set[tuple[int, int]] = set()

    def add_pair(a: int, b: int, band: str) -> bool:
        if a == b:
            return False
        key = (min(a, b), max(a, b))
        if key in seen_pairs:
            return False
        seen_pairs.add(key)
        pairs.append(CalibrationPair(segment_a_id=a, segment_b_id=b, band=band))
        return True

    # High + medium bands: draw from real retrieval results for random query segments.
    query_order = list(rng.permutation(len(all_seg_ids)))
    high_count = medium_count = 0
    for idx in query_order:
        if high_count >= n_high and medium_count >= n_medium:
            break
        seg_id = all_seg_ids[idx]
        song_id = segment_song[seg_id]
        query_vec = embedding_repo.get_vector(facet_name, seg_id)
        raw = embedding_repo.search(facet_name, query_vec, k=20)
        neighbors = [cid for cid, _ in raw if segment_song.get(cid) not in (None, song_id)]
        if high_count < n_high and neighbors:
            if add_pair(seg_id, neighbors[0], "high"):
                high_count += 1
        if medium_count < n_medium and len(neighbors) >= 10:
            if add_pair(seg_id, neighbors[9], "medium"):
                medium_count += 1

    # Random band: uniformly random cross-song pairs.
    random_count = 0
    attempts = 0
    max_attempts = max(1, n_random) * 20
    while random_count < n_random and attempts < max_attempts and len(all_seg_ids) >= 2:
        attempts += 1
        a, b = rng.choice(len(all_seg_ids), size=2, replace=False)
        seg_a, seg_b = all_seg_ids[a], all_seg_ids[b]
        if segment_song[seg_a] == segment_song[seg_b]:
            continue
        if add_pair(seg_a, seg_b, "random"):
            random_count += 1

    order = rng.permutation(len(pairs))  # bands shouldn't be presentable in generation order
    return [pairs[i] for i in order]
