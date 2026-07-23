"""Wraps calibration_ratings (SQLite) -- human XAB similarity judgments, the
dataset section 9's fine-tuning/blend-weight-regression work depends on. Each
row is one triplet (reference X, candidates A/B) plus which candidate the
rater judged more similar to X. No UNIQUE constraint on (x, a, b) in the
schema -- callers that want "already rated" semantics (the rating tool)
check get_all_ratings() themselves rather than relying on an upsert; a
triplet genuinely could be shown to more than one rater someday."""

import sqlite3


class CalibrationRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add_choice(
        self, segment_x_id: int, segment_a_id: int, segment_b_id: int, choice: str, rater: str | None = None
    ) -> int:
        if choice not in ("a", "b"):
            raise ValueError(f"choice must be 'a' or 'b', got {choice!r}")
        cur = self.conn.execute(
            "INSERT INTO calibration_ratings (segment_x_id, segment_a_id, segment_b_id, choice, rater) "
            "VALUES (?, ?, ?, ?, ?)",
            (segment_x_id, segment_a_id, segment_b_id, choice, rater),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_all_ratings(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, segment_x_id, segment_a_id, segment_b_id, choice, rater, created_at "
            "FROM calibration_ratings"
        ).fetchall()

    def rated_triplet_keys(self) -> set[tuple[int, int, int]]:
        """(segment_x_id, min(a,b), max(a,b)) for every already-rated triplet --
        used to skip triplets a rating tool has already shown, across
        sessions. Candidate order (which one was "A" vs "B") is randomized at
        generation time to avoid position bias, so identity for dedup
        purposes ignores which slot each candidate landed in."""
        keys = set()
        for row in self.get_all_ratings():
            a, b = row["segment_a_id"], row["segment_b_id"]
            keys.add((row["segment_x_id"], min(a, b), max(a, b)))
        return keys

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM calibration_ratings").fetchone()
        return row["n"]
