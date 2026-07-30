"""Wraps taste_ratings / taste_ratings_guest (SQLite) -- "do I like this
segment," a deliberately separate judgment from CalibrationRepository's
similarity XAB (see db.py's schema comment). One row per segment, liked or
not, no comparison involved.

table picks which physical table this instance reads/writes -- same
real-vs-guest, separate-table-not-a-flag reasoning as CalibrationRepository."""

import sqlite3

_ALLOWED_TABLES = ("taste_ratings", "taste_ratings_guest")


class TasteRepository:
    def __init__(self, conn: sqlite3.Connection, table: str = "taste_ratings"):
        if table not in _ALLOWED_TABLES:
            raise ValueError(f"table must be one of {_ALLOWED_TABLES}, got {table!r}")
        self.conn = conn
        self.table = table

    def add_rating(self, segment_id: int, liked: bool, rater: str | None = None) -> int:
        cur = self.conn.execute(
            f"INSERT INTO {self.table} (segment_id, liked, rater) VALUES (?, ?, ?)",
            (segment_id, int(liked), rater),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_all_ratings(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            f"SELECT id, segment_id, liked, rater, created_at FROM {self.table}"
        ).fetchall()

    def rated_segment_ids(self, rater: str | None = None) -> set[int]:
        """Segment ids already rated -- used to skip segments a rating tool
        has already shown. rater, when given, restricts this to one
        profile's own history, same reasoning as CalibrationRepository.
        rated_triplet_keys: each profile's taste is independent, so one
        profile rating a segment must not remove it from another profile's
        queue."""
        query = f"SELECT segment_id FROM {self.table}"
        params: tuple = ()
        if rater is not None:
            query += " WHERE rater = ?"
            params = (rater,)
        return {row["segment_id"] for row in self.conn.execute(query, params)}

    def count(self) -> int:
        row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {self.table}").fetchone()
        return row["n"]
