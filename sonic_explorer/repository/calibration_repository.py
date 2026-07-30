"""Wraps calibration_ratings / calibration_ratings_guest (SQLite) -- human XAB
similarity judgments, the dataset section 9's fine-tuning/blend-weight-
regression work depends on. Each row is one triplet (reference X, candidates
A/B) plus which candidate the rater judged more similar to X. No UNIQUE
constraint on (x, a, b) in the schema -- callers that want "already rated"
semantics (the rating tool) check get_all_ratings()/rated_triplet_keys()
themselves rather than relying on an upsert; a triplet genuinely could be
shown to more than one rater by design now (multiple calibration profiles).

table picks which physical table this instance reads/writes -- real profiles
go to calibration_ratings, the "Guest / Test" profile goes to
calibration_ratings_guest, a separate table (not a flag column) specifically
so guest data can never accidentally end up in the blend-weight regression;
see db.py's schema comment for the full reasoning. Validated against a fixed
allow-list before use in a query, even though callers are internal/trusted,
since it's interpolated into SQL rather than bound as a parameter (SQLite
doesn't support parameterized table names)."""

import sqlite3

_ALLOWED_TABLES = ("calibration_ratings", "calibration_ratings_guest")


class CalibrationRepository:
    def __init__(self, conn: sqlite3.Connection, table: str = "calibration_ratings"):
        if table not in _ALLOWED_TABLES:
            raise ValueError(f"table must be one of {_ALLOWED_TABLES}, got {table!r}")
        self.conn = conn
        self.table = table

    def add_choice(
        self,
        segment_x_id: int,
        segment_a_id: int,
        segment_b_id: int,
        choice: str,
        rater: str | None = None,
        sampling_facet: str | None = None,
    ) -> int:
        """sampling_facet records which facet's retrieval results picked
        this triplet's candidates -- "sound" for the main pool, "harmony"
        (etc.) for a supplemental sanity-check pool sampled differently.
        None (the default) means the caller didn't specify -- treated the
        same as "sound" by every existing row and by convention (see db.py's
        migration comment), not left ambiguous on purpose."""
        if choice not in ("a", "b"):
            raise ValueError(f"choice must be 'a' or 'b', got {choice!r}")
        cur = self.conn.execute(
            f"INSERT INTO {self.table} "
            "(segment_x_id, segment_a_id, segment_b_id, choice, rater, sampling_facet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (segment_x_id, segment_a_id, segment_b_id, choice, rater, sampling_facet),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_all_ratings(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            f"SELECT id, segment_x_id, segment_a_id, segment_b_id, choice, rater, sampling_facet, created_at "
            f"FROM {self.table}"
        ).fetchall()

    def rated_triplet_keys(self, rater: str | None = None) -> set[tuple[int, int, int]]:
        """(segment_x_id, min(a,b), max(a,b)) for every already-rated triplet --
        used to skip triplets a rating tool has already shown, across
        sessions. Candidate order (which one was "A" vs "B") is randomized at
        generation time to avoid position bias, so identity for dedup
        purposes ignores which slot each candidate landed in.

        rater, when given, restricts this to one profile's own history --
        essential now that multiple real profiles calibrate independently:
        without this filter, whichever profile rates a triplet first would
        make it disappear for every other profile too, instead of each
        profile seeing (and needing to rate) the full independent set."""
        query = f"SELECT segment_x_id, segment_a_id, segment_b_id FROM {self.table}"
        params: tuple = ()
        if rater is not None:
            query += " WHERE rater = ?"
            params = (rater,)
        keys = set()
        for row in self.conn.execute(query, params):
            a, b = row["segment_a_id"], row["segment_b_id"]
            keys.add((row["segment_x_id"], min(a, b), max(a, b)))
        return keys

    def count(self) -> int:
        row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {self.table}").fetchone()
        return row["n"]
