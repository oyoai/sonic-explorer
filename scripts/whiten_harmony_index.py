"""One-off: re-whitens the harmony facet's FAISS index in place, and prints
a before/after comparison. See sonic_explorer/analysis/embedding_whitening.py
for why -- retrieval_diagnostics.py found harmony's raw chroma-derived
vectors have very little natural spread (random-pair cosine similarity
already sits at 0.85-0.95), collapsing the space genre-cohesion needs to
discriminate on.

Purely a post-hoc transform on vectors already computed and indexed -- no
audio re-processing, no re-embedding. Backs up the original index first
(harmony.index.pre_whitening.bak) so this is reversible. Meant to run once;
re-running would fit a whitener on an already-whitened corpus (mean ~0, std
~1 already), which is a near-identity transform, not harmful but pointless.
"""

import shutil

from sonic_explorer.analysis.embedding_whitening import fit_whitener
from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH
from sonic_explorer.evaluation.genre_cohesion import genre_cohesion_at_k
from sonic_explorer.evaluation.retrieval_diagnostics import top1_score_distribution
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

FACET_NAME = "harmony"
SAMPLE_SIZE = 300


def _measure(song_repo, embedding_repo, label):
    scores = top1_score_distribution(song_repo, embedding_repo, facet_name=FACET_NAME, sample_size=SAMPLE_SIZE)
    cohesion = genre_cohesion_at_k(song_repo, embedding_repo, facet_name=FACET_NAME, k=10, sample_size=SAMPLE_SIZE)
    import numpy as np
    print(f"\n--- {label} ---")
    print(f"  top-1 score:        mean={np.mean(scores.top1_scores):.3f}")
    print(f"  random-pair score:  mean={np.mean(scores.random_pair_scores):.3f}")
    print(f"  top1-vs-random gap: {np.mean(scores.top1_scores) - np.mean(scores.random_pair_scores):.3f}")
    print(f"  top1-vs-top2 margin: mean={np.mean(scores.top1_top2_margins):.4f}")
    print(f"  genre-cohesion@10:   {cohesion.observed * 100:.1f}% (random baseline {cohesion.random_baseline * 100:.1f}%)")
    return {
        "top1_mean": float(np.mean(scores.top1_scores)),
        "random_mean": float(np.mean(scores.random_pair_scores)),
        "margin_mean": float(np.mean(scores.top1_top2_margins)),
        "genre_cohesion_pct": cohesion.observed * 100,
        "genre_cohesion_baseline_pct": cohesion.random_baseline * 100,
    }


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
    embedding_repo.load_index(FACET_NAME)

    before = _measure(song_repo, embedding_repo, "BEFORE whitening")

    index_path = embedding_repo._index_path(FACET_NAME)
    backup_path = index_path.with_suffix(".index.pre_whitening.bak")
    shutil.copy(index_path, backup_path)
    print(f"\nBacked up original index to {backup_path}")

    # Pull every currently-indexed vector, fit + apply the whitener, rebuild fresh.
    segment_ids = []
    for song in song_repo.list_songs():
        for seg in song_repo.get_segments(song.id):
            if embedding_repo.status(seg.id, FACET_NAME) == "done":
                segment_ids.append(seg.id)

    vectors = [embedding_repo.get_vector(FACET_NAME, seg_id) for seg_id in segment_ids]
    whitener = fit_whitener(vectors)
    whitened = [whitener.transform(v) for v in vectors]

    import faiss
    import numpy as np
    dim = whitened[0].shape[0]
    new_index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    new_index.add_with_ids(np.stack(whitened).astype(np.float32), np.array(segment_ids, dtype=np.int64))
    embedding_repo._indexes[FACET_NAME] = new_index
    embedding_repo.save_index(FACET_NAME)
    print(f"Rebuilt {FACET_NAME} index with {len(segment_ids)} whitened vectors.")

    # Reload fresh from disk to measure exactly what's now persisted.
    embedding_repo2 = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
    embedding_repo2.load_index(FACET_NAME)
    after = _measure(song_repo, embedding_repo2, "AFTER whitening")

    print("\n=== Summary ===")
    print(f"top1-vs-random gap: {before['top1_mean'] - before['random_mean']:.3f} -> "
          f"{after['top1_mean'] - after['random_mean']:.3f}")
    print(f"genre-cohesion@10:  {before['genre_cohesion_pct']:.1f}% -> {after['genre_cohesion_pct']:.1f}%")


if __name__ == "__main__":
    main()
