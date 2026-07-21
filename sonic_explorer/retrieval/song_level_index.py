"""Song-level FAISS indexes -- one mean-pooled vector per song per facet,
built from mean_pool_song_vectors() (the same aggregation Taste Map/Explore
already use for visualization, now reused for retrieval).

Why: retrieval_diagnostics.py's score-distribution check found a near-flat
top1-vs-top2 margin across every facet (typically <0.01) at the segment
level -- with ~14,600 segments and often only a few hundred per genre,
there's usually a long plateau of near-tied single-segment candidates rather
than one clearly-best match. Aggregating a song's segments into one vector
before ranking smooths that segment-level noise into a sharper song-level
signal -- at the real cost of losing moment-level granularity (a result is
"this song," not "this specific 5s within it")."""

import faiss
import numpy as np

from sonic_explorer.analysis.taste_map import mean_pool_song_vectors


def build_song_level_index(song_repo, embedding_repo, facet_name: str) -> faiss.IndexIDMap2 | None:
    """None if no songs have this facet embedded yet."""
    song_vectors = mean_pool_song_vectors(song_repo, embedding_repo, facet_name=facet_name)
    if not song_vectors:
        return None

    song_ids = list(song_vectors.keys())
    matrix = np.stack([song_vectors[sid] for sid in song_ids]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(matrix.shape[1]))
    index.add_with_ids(matrix, np.array(song_ids, dtype=np.int64))
    return index


def query_song_level(
    index: faiss.IndexIDMap2, query_vector: np.ndarray, k: int = 10, exclude_song_id: int | None = None
) -> list[tuple[int, float]]:
    """Returns [(song_id, score), ...], best first."""
    vec = query_vector.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    fetch_k = min(k + 1 if exclude_song_id is not None else k, index.ntotal)
    if fetch_k <= 0:
        return []

    scores, ids = index.search(vec.reshape(1, -1), fetch_k)
    results = [(int(sid), float(score)) for sid, score in zip(ids[0], scores[0]) if sid != -1]
    if exclude_song_id is not None:
        results = [r for r in results if r[0] != exclude_song_id]
    return results[:k]
