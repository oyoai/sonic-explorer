"""Generates XAB triplets (reference X, candidates A/B) for human similarity
judgment -- the calibration_ratings dataset section 9's blend-weight
regression and (conditionally) CLAP fine-tuning depend on. XAB (pick which
of A/B sounds more like X) rather than a raw 1-5 scale, per Vohra & Akama
(2026)'s ABX-preference-based methodology (see Overview's Related Work) --
a forced binary discrimination is a more rigorous, less subjective task than
an absolute similarity rating, and it's what the paper this design borrows
from actually validated.

Deliberately not built from naive random candidates, for the same reason the
earlier pair-based version wasn't: a naive random pair skews almost entirely
to "obviously dissimilar" (see evaluation/retrieval_diagnostics.py's
random-pair-baseline numbers), giving little real variance to learn from.
Each triplet's two candidates are instead drawn from two of three bands off
the reference's real retrieval results -- high similarity (real top-1),
medium (rank ~10), and random -- rotating through all three pairwise band
combinations across the generated set, so a rater is always making a
genuine, non-trivial discrimination rather than an obvious call."""

from dataclasses import dataclass
from itertools import cycle

import numpy as np

from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

_BAND_PAIR_ROTATION = [("high", "medium"), ("high", "random"), ("medium", "random")]


@dataclass
class CalibrationTriplet:
    segment_x_id: int  # reference
    segment_a_id: int  # candidate A
    segment_b_id: int  # candidate B
    band_a: str  # "high" | "medium" | "random" -- for reporting spread, never shown to the rater
    band_b: str


def _random_cross_song_segment(rng, all_seg_ids: list[int], segment_song: dict[int, int], exclude_song_id: int) -> int | None:
    for _ in range(20):
        candidate = all_seg_ids[rng.integers(len(all_seg_ids))]
        if segment_song[candidate] != exclude_song_id:
            return candidate
    return None


def generate_calibration_triplets(
    song_repo: SongRepository,
    embedding_repo: EmbeddingRepository,
    facet_name: str = "sound",
    n_triplets: int = 350,
    seed: int = 42,
) -> list[CalibrationTriplet]:
    rng = np.random.default_rng(seed)
    songs = song_repo.list_songs()

    segment_song: dict[int, int] = {}
    for song in songs:
        for seg in song_repo.get_segments(song.id):
            if embedding_repo.status(seg.id, facet_name) == "done":
                segment_song[seg.id] = song.id

    all_seg_ids = list(segment_song.keys())
    if len(all_seg_ids) < 3:
        return []

    triplets: list[CalibrationTriplet] = []
    seen: set[tuple[int, int, int]] = set()

    query_order = list(rng.permutation(len(all_seg_ids)))
    band_pairs = cycle(_BAND_PAIR_ROTATION)

    for idx in query_order:
        if len(triplets) >= n_triplets:
            break
        x_id = all_seg_ids[idx]
        x_song_id = segment_song[x_id]
        band_a, band_b = next(band_pairs)

        query_vec = embedding_repo.get_vector(facet_name, x_id)
        raw = embedding_repo.search(facet_name, query_vec, k=20)
        neighbors = [cid for cid, _ in raw if segment_song.get(cid) not in (None, x_song_id)]

        candidates: dict[str, int] = {}
        if neighbors:
            candidates["high"] = neighbors[0]
        if len(neighbors) >= 10:
            candidates["medium"] = neighbors[9]
        random_candidate = _random_cross_song_segment(rng, all_seg_ids, segment_song, exclude_song_id=x_song_id)
        if random_candidate is not None:
            candidates["random"] = random_candidate

        if band_a not in candidates or band_b not in candidates:
            continue  # this query lacks one of the requested bands -- skip, try the next query
        seg_a, seg_b = candidates[band_a], candidates[band_b]
        if seg_a == seg_b:
            continue

        key = (x_id, min(seg_a, seg_b), max(seg_a, seg_b))
        if key in seen:
            continue
        seen.add(key)

        # Randomize which candidate lands in slot A vs B -- otherwise the
        # "closer" candidate would always land in the same slot, and a rater
        # could learn to just always pick one side without listening.
        if rng.random() < 0.5:
            triplets.append(CalibrationTriplet(x_id, seg_a, seg_b, band_a, band_b))
        else:
            triplets.append(CalibrationTriplet(x_id, seg_b, seg_a, band_b, band_a))

    order = rng.permutation(len(triplets))
    return [triplets[i] for i in order]
