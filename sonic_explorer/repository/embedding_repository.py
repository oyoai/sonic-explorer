"""Wraps embedding_status (SQLite) + one FAISS index per facet. The only place
faiss.* calls happen anywhere in the codebase. Implements the compute-once pattern:
check status() before embedding a segment; skip if already 'done'."""

import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np

from sonic_explorer.config import ARTIFACTS_DIR


class EmbeddingRepository:
    def __init__(self, conn: sqlite3.Connection, artifacts_dir: str | Path | None = None):
        """artifacts_dir defaults to the package-local data/artifacts/ (fine for local
        dev/tests), but Colab callers must pass the Drive-mounted artifacts folder --
        otherwise the FAISS index would only exist on the ephemeral /content disk and
        vanish on disconnect, defeating the whole compute-once/resumability point."""
        self.conn = conn
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
        self._indexes: dict[str, faiss.IndexIDMap2] = {}

    def _index_path(self, facet_name: str) -> Path:
        return self.artifacts_dir / f"{facet_name}.index"

    def _get_or_create_index(self, facet_name: str, dim: int) -> faiss.IndexIDMap2:
        if facet_name not in self._indexes:
            self._indexes[facet_name] = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        return self._indexes[facet_name]

    def status(self, segment_id: int, facet_name: str) -> str:
        row = self.conn.execute(
            "SELECT status FROM embedding_status WHERE segment_id = ? AND facet_name = ?",
            (segment_id, facet_name),
        ).fetchone()
        return row["status"] if row else "pending"

    def add_vector(self, facet_name: str, segment_id: int, vector: np.ndarray) -> int:
        """Adds to the facet's FAISS index (id = segment_id) and marks embedding_status 'done'.
        Returns the vector_store_id (== segment_id, since each facet has its own index)."""
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        index = self._get_or_create_index(facet_name, len(vec))
        index.add_with_ids(vec.reshape(1, -1), np.array([segment_id], dtype=np.int64))
        self.mark_done(segment_id, facet_name, vector_store_id=segment_id, dim=len(vec))
        return segment_id

    def mark_done(self, segment_id: int, facet_name: str, vector_store_id: int, dim: int | None = None) -> None:
        self.conn.execute(
            "INSERT INTO embedding_status (segment_id, facet_name, status, vector_store_id, dim, computed_at) "
            "VALUES (?, ?, 'done', ?, ?, ?) "
            "ON CONFLICT(segment_id, facet_name) DO UPDATE SET "
            "status='done', vector_store_id=excluded.vector_store_id, dim=excluded.dim, computed_at=excluded.computed_at",
            (segment_id, facet_name, vector_store_id, dim, time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self.conn.commit()

    def search(self, facet_name: str, query_vector: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if facet_name not in self._indexes or self._indexes[facet_name].ntotal == 0:
            return []
        vec = query_vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        index = self._indexes[facet_name]
        k = min(k, index.ntotal)
        scores, ids = index.search(vec.reshape(1, -1), k)
        return [(int(seg_id), float(score)) for seg_id, score in zip(ids[0], scores[0]) if seg_id != -1]

    def index_size(self, facet_name: str) -> int:
        return self._indexes[facet_name].ntotal if facet_name in self._indexes else 0

    def get_vector(self, facet_name: str, segment_id: int) -> np.ndarray:
        """Fetch a segment's already-computed vector back out of the index --
        avoids ever needing to reload audio + re-embed for a segment we've already
        processed (used by retrieval for query-by-segment, and for sanity checks).
        FAISS's SWIG binding rejects numpy int types (e.g. from rng.choice(...) or
        a pandas row) -- cast to plain int defensively."""
        return np.array(self._indexes[facet_name].reconstruct(int(segment_id)), dtype=np.float32)

    def save_index(self, facet_name: str) -> None:
        if facet_name not in self._indexes:
            return
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._indexes[facet_name], str(self._index_path(facet_name)))

    def load_index(self, facet_name: str) -> None:
        path = self._index_path(facet_name)
        if path.exists():
            self._indexes[facet_name] = faiss.read_index(str(path))

    def get_structure_matrix(self, song_id: int) -> np.ndarray:
        """Song-level self-similarity matrix -- the artifact Song X-Ray reads directly,
        outside the facet-vector retrieval path (see structure facet notes, Day 3)."""
        return np.load(self.artifacts_dir / "structure" / f"{song_id}.npy")
