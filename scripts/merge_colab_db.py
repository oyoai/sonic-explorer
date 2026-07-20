"""Merges embedding_status rows for specific facets from a downloaded Colab
DB into the local DB -- for when local-only work (e.g. harmony/structure,
computed after a Colab sync) has diverged from a later Colab run's starting
DB snapshot, so a plain overwrite would silently discard one side's work.

Only touches embedding_status rows for the given facets -- songs, segments,
song DNA, and every other facet's status in the local DB are left alone.
Refuses to merge if song/segment counts don't match between the two DBs
(the merge is by segment_id, which is only safe if both DBs trace back to
the exact same songs/segments having never diverged).

Backs up the local DB (sonic_explorer.db.bak) before writing, since this is
a one-shot manual-recovery script, not a routine pipeline step -- worth the
extra safety margin.

Usage: python scripts/merge_colab_db.py <path-to-downloaded-colab-db> [facet ...]
       (facets default to vocal, drums, bass, instrumental)
"""

import shutil
import sqlite3
import sys
from pathlib import Path

from sonic_explorer.config import DB_PATH

DEFAULT_FACETS = ["vocal", "drums", "bass", "instrumental"]


def main(colab_db_path: str, facets: list[str]):
    colab_path = Path(colab_db_path)
    if not colab_path.exists():
        raise SystemExit(f"Colab DB not found at {colab_path}")
    if not DB_PATH.exists():
        raise SystemExit(f"Local DB not found at {DB_PATH}")

    backup_path = DB_PATH.with_suffix(".db.bak")
    shutil.copy(DB_PATH, backup_path)
    print(f"Backed up local DB to {backup_path}")

    local_conn = sqlite3.connect(str(DB_PATH))
    colab_conn = sqlite3.connect(str(colab_path))
    colab_conn.row_factory = sqlite3.Row

    local_songs = local_conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    colab_songs = colab_conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    if local_songs != colab_songs:
        raise SystemExit(
            f"Song count mismatch: local={local_songs}, colab={colab_songs} -- "
            "refusing to merge by segment_id, this needs manual review."
        )

    local_segs = local_conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    colab_segs = colab_conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    if local_segs != colab_segs:
        raise SystemExit(
            f"Segment count mismatch: local={local_segs}, colab={colab_segs} -- "
            "refusing to merge by segment_id, this needs manual review."
        )
    print(f"Sanity check passed: {local_songs} songs, {local_segs} segments match on both sides.")

    for facet_name in facets:
        rows = colab_conn.execute(
            "SELECT segment_id, status, vector_store_id, dim, computed_at "
            "FROM embedding_status WHERE facet_name = ?",
            (facet_name,),
        ).fetchall()
        for row in rows:
            local_conn.execute(
                "INSERT INTO embedding_status (segment_id, facet_name, status, vector_store_id, dim, computed_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(segment_id, facet_name) DO UPDATE SET "
                "status=excluded.status, vector_store_id=excluded.vector_store_id, "
                "dim=excluded.dim, computed_at=excluded.computed_at",
                (row["segment_id"], facet_name, row["status"], row["vector_store_id"], row["dim"], row["computed_at"]),
            )
        local_conn.commit()
        done = sum(1 for r in rows if r["status"] == "done")
        skipped = sum(1 for r in rows if r["status"] == "skipped")
        print(f"{facet_name}: merged {len(rows)} rows ({done} done, {skipped} skipped)")

    local_conn.close()
    colab_conn.close()
    print("Merge complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    colab_db_arg = sys.argv[1]
    facets_arg = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_FACETS
    main(colab_db_arg, facets_arg)
