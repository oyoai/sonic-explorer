"""One-off repair for a real bug in the pre-fix run_batch_embedding: a Colab
disconnect between FAISS index checkpoints could leave segments marked 'done' in
the DB with no corresponding vector actually persisted in the saved index (mark_done
fired immediately per-vector, decoupled from when save_index next ran). See
sonic_explorer/pipeline/embed_library.py and EmbeddingRepository.add_to_index()
for the fix -- this script only repairs data already affected by the old bug.

Finds every 'done' segment whose vector isn't actually reconstructable from the
FAISS index and resets it back to 'pending', so downstream code (mean_pool_song_vectors,
Moment Matcher's status guard, etc.) correctly skips it instead of crashing.
Safe to re-run.
"""

import faiss

from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

FACET_NAME = "sound"


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
    embedding_repo.load_index(FACET_NAME)

    index = embedding_repo._indexes.get(FACET_NAME)
    index_ids = set(faiss.vector_to_array(index.id_map).tolist()) if index is not None else set()

    rows = conn.execute(
        "SELECT segment_id FROM embedding_status WHERE facet_name = ? AND status = 'done'", (FACET_NAME,)
    ).fetchall()
    db_done_ids = [r["segment_id"] for r in rows]

    orphaned = [seg_id for seg_id in db_done_ids if seg_id not in index_ids]

    for seg_id in orphaned:
        embedding_repo.reset_status(seg_id, FACET_NAME)

    print(f"Checked {len(db_done_ids)} 'done' rows against {len(index_ids)} vectors in the FAISS index")
    print(f"Reset {len(orphaned)} orphaned segments back to 'pending'")

    if orphaned:
        affected_songs = set()
        for seg_id in orphaned:
            row = conn.execute("SELECT song_id FROM segments WHERE id = ?", (seg_id,)).fetchone()
            if row:
                affected_songs.add(row["song_id"])
        print(f"Affected songs: {len(affected_songs)} (now missing some/all sound-facet coverage): "
              f"{sorted(affected_songs)}")


if __name__ == "__main__":
    main()
