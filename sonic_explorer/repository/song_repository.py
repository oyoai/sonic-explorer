"""Wraps all songs/segments SQL. Nothing outside this file issues raw SQL for these tables."""

import sqlite3

from sonic_explorer.models import Segment, Song


class SongRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add_song(self, song: Song) -> int:
        """Idempotent on fma_track_id -- re-running the batch pipeline never duplicates songs."""
        existing = self.conn.execute(
            "SELECT id FROM songs WHERE fma_track_id = ?", (song.fma_track_id,)
        ).fetchone()
        if existing:
            return existing["id"]
        cur = self.conn.execute(
            "INSERT INTO songs (fma_track_id, filepath, title, artist, genre_top, duration_sec) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (song.fma_track_id, song.filepath, song.title, song.artist, song.genre_top, song.duration_sec),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_filepath(self, song_id: int, filepath: str) -> None:
        self.conn.execute("UPDATE songs SET filepath = ? WHERE id = ?", (filepath, song_id))
        self.conn.commit()

    def update_song_dna(
        self, song_id: int, tempo_bpm: float, energy: float, brightness: float,
        harmonic_complexity: float, rhythmic_density: float,
    ) -> None:
        self.conn.execute(
            "UPDATE songs SET tempo_bpm = ?, energy = ?, brightness = ?, "
            "harmonic_complexity = ?, rhythmic_density = ? WHERE id = ?",
            (tempo_bpm, energy, brightness, harmonic_complexity, rhythmic_density, song_id),
        )
        self.conn.commit()

    @staticmethod
    def _song_from_row(row) -> Song:
        return Song(
            id=row["id"],
            fma_track_id=row["fma_track_id"],
            filepath=row["filepath"],
            title=row["title"],
            artist=row["artist"],
            genre_top=row["genre_top"],
            duration_sec=row["duration_sec"],
            tempo_bpm=row["tempo_bpm"],
            energy=row["energy"],
            brightness=row["brightness"],
            harmonic_complexity=row["harmonic_complexity"],
            rhythmic_density=row["rhythmic_density"],
            is_saved=bool(row["is_saved"]),
            description=row["description"],
        )

    def update_description(self, song_id: int, description: str) -> None:
        self.conn.execute("UPDATE songs SET description = ? WHERE id = ?", (description, song_id))
        self.conn.commit()

    def save_song(self, song_id: int) -> None:
        self.conn.execute("UPDATE songs SET is_saved = 1 WHERE id = ?", (song_id,))
        self.conn.commit()

    def unsave_song(self, song_id: int) -> None:
        self.conn.execute("UPDATE songs SET is_saved = 0 WHERE id = ?", (song_id,))
        self.conn.commit()

    def get_song(self, song_id: int) -> Song | None:
        row = self.conn.execute("SELECT * FROM songs WHERE id = ?", (song_id,)).fetchone()
        if row is None:
            return None
        song = self._song_from_row(row)
        song.segments = self.get_segments(song_id)
        return song

    def get_song_by_fma_track_id(self, fma_track_id: int) -> Song | None:
        row = self.conn.execute("SELECT id FROM songs WHERE fma_track_id = ?", (fma_track_id,)).fetchone()
        if row is None:
            return None
        return self.get_song(row["id"])

    def list_songs(self, genre: str | None = None, saved_only: bool = False) -> list[Song]:
        clauses, params = [], []
        if genre is not None:
            clauses.append("genre_top = ?")
            params.append(genre)
        if saved_only:
            clauses.append("is_saved = 1")
        query = "SELECT * FROM songs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        rows = self.conn.execute(query, params).fetchall()
        return [self._song_from_row(row) for row in rows]

    def add_segments(self, song_id: int, segments: list[Segment]) -> list[int]:
        """Idempotent on (song_id, start_sec, end_sec) -- safe to re-run."""
        ids = []
        for seg in segments:
            existing = self.conn.execute(
                "SELECT id FROM segments WHERE song_id = ? AND start_sec = ? AND end_sec = ?",
                (song_id, seg.start_sec, seg.end_sec),
            ).fetchone()
            if existing:
                ids.append(existing["id"])
                continue
            cur = self.conn.execute(
                "INSERT INTO segments (song_id, start_sec, end_sec, segment_index) VALUES (?, ?, ?, ?)",
                (song_id, seg.start_sec, seg.end_sec, seg.segment_index),
            )
            ids.append(cur.lastrowid)
        self.conn.commit()
        return ids

    def get_segments(self, song_id: int) -> list[Segment]:
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE song_id = ? ORDER BY segment_index", (song_id,)
        ).fetchall()
        return [
            Segment(
                id=row["id"],
                song_id=row["song_id"],
                start_sec=row["start_sec"],
                end_sec=row["end_sec"],
                segment_index=row["segment_index"],
            )
            for row in rows
        ]

    def get_segment(self, segment_id: int) -> Segment | None:
        row = self.conn.execute("SELECT * FROM segments WHERE id = ?", (segment_id,)).fetchone()
        if row is None:
            return None
        return Segment(
            id=row["id"],
            song_id=row["song_id"],
            start_sec=row["start_sec"],
            end_sec=row["end_sec"],
            segment_index=row["segment_index"],
        )
