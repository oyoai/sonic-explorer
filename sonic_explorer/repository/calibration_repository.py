"""Wraps calibration_ratings (SQLite) -- human similarity ratings on segment
pairs, the dataset section 9's fine-tuning/blend-weight-regression work
depends on. No UNIQUE constraint on (segment_a_id, segment_b_id) in the
schema -- callers that want "already rated" semantics (the rating tool)
check get_all_ratings() themselves rather than relying on an upsert; a
segment pair genuinely could be shown to more than one rater someday."""

import sqlite3


class CalibrationRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add_rating(self, segment_a_id: int, segment_b_id: int, rating: float, rater: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO calibration_ratings (segment_a_id, segment_b_id, rating, rater) VALUES (?, ?, ?, ?)",
            (segment_a_id, segment_b_id, rating, rater),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_all_ratings(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, segment_a_id, segment_b_id, rating, rater, created_at FROM calibration_ratings"
        ).fetchall()

    def rated_pair_ids(self) -> set[tuple[int, int]]:
        """(segment_a_id, segment_b_id) pairs already rated, both orderings --
        used to skip pairs a rating tool has already shown, across sessions."""
        pairs = set()
        for row in self.get_all_ratings():
            pairs.add((row["segment_a_id"], row["segment_b_id"]))
            pairs.add((row["segment_b_id"], row["segment_a_id"]))
        return pairs

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM calibration_ratings").fetchone()
        return row["n"]
