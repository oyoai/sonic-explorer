"""SQLite schema (spec 8.1's four tables) and connection helper.
No raw SQL happens anywhere outside this repository/ package."""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fma_track_id INTEGER UNIQUE NOT NULL,
    filepath TEXT NOT NULL,
    title TEXT,
    artist TEXT,
    genre_top TEXT,
    duration_sec REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id INTEGER NOT NULL REFERENCES songs(id),
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    segment_index INTEGER NOT NULL,
    UNIQUE(song_id, start_sec, end_sec)
);

CREATE TABLE IF NOT EXISTS embedding_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id INTEGER NOT NULL REFERENCES segments(id),
    facet_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    vector_store_id INTEGER,
    dim INTEGER,
    computed_at TEXT,
    UNIQUE(segment_id, facet_name)
);

CREATE TABLE IF NOT EXISTS calibration_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_a_id INTEGER NOT NULL REFERENCES segments(id),
    segment_b_id INTEGER NOT NULL REFERENCES segments(id),
    rating REAL NOT NULL,
    rater TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    # check_same_thread=False: callers that cache this connection long-lived (e.g.
    # Streamlit's @st.cache_resource) will see it reused across the framework's
    # script-rerun thread pool -- reruns are sequential per session, not truly
    # concurrent, so this is safe for our access pattern.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
