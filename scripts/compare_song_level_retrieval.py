"""One-off experiment: does song-level aggregation (retrieval/song_level_index.py)
actually sharpen retrieval quality vs. segment-level, per retrieval_diagnostics.py's
finding that segment-level top1-vs-top2 margins are near-flat across every facet?
Compares both on the same two metrics (score-distribution margins, genre-cohesion@10)
for all six facets. Read-only -- doesn't modify any persisted data, just reports."""

import numpy as np

from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH
from sonic_explorer.evaluation.genre_cohesion import genre_cohesion_at_k, song_level_genre_cohesion_at_k
from sonic_explorer.evaluation.retrieval_diagnostics import song_level_score_distribution, top1_score_distribution
from sonic_explorer.facets.registry import default_registry
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

K = 10
SAMPLE_SIZE = 300


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
    facets = default_registry().names()
    for f in facets:
        embedding_repo.load_index(f)

    print(f"{'facet':14s} {'seg margin':>11s} {'song margin':>12s} {'seg cohesion':>13s} {'song cohesion':>14s}")
    for facet_name in facets:
        seg_scores = top1_score_distribution(song_repo, embedding_repo, facet_name=facet_name, sample_size=SAMPLE_SIZE)
        song_scores = song_level_score_distribution(song_repo, embedding_repo, facet_name=facet_name, sample_size=SAMPLE_SIZE)
        seg_cohesion = genre_cohesion_at_k(song_repo, embedding_repo, facet_name=facet_name, k=K, sample_size=SAMPLE_SIZE)
        song_cohesion = song_level_genre_cohesion_at_k(song_repo, embedding_repo, facet_name=facet_name, k=K, sample_size=SAMPLE_SIZE)

        seg_margin = np.mean(seg_scores.top1_top2_margins) if seg_scores.top1_top2_margins else float("nan")
        song_margin = np.mean(song_scores.top1_top2_margins) if song_scores.top1_top2_margins else float("nan")

        print(f"{facet_name:14s} {seg_margin:11.4f} {song_margin:12.4f} "
              f"{seg_cohesion.observed * 100:12.1f}% {song_cohesion.observed * 100:13.1f}%")


if __name__ == "__main__":
    main()
