"""One-off filter pass over the vocal facet: for every 'done' vocal-facet
segment, run a pretrained AudioSet tagger (AST) against that SPECIFIC
segment's audio (not the whole song) to check whether it actually contains
singing/speech (pipeline/vocal_presence.py). Segments that fail the check
are removed from the FAISS index and marked 'skipped' individually.

Segment-level, not song-level, by design: a real vocal song can still have
instrumental-only stretches (intro, bridge, break) that correctly score low
on vocal presence -- that's the model being accurate, not a bug. Moment
Matcher queries a specific moment and expects similar moments back, so an
instrumental bridge from an otherwise-vocal song must not be excluded or
included as a block just because the rest of its song has real vocals. An
earlier whole-song design (score once, gate the whole song) also failed
validation outright (see pipeline/vocal_presence.py's docstring) -- this
isn't just a granularity preference, it's the only design that validated.

Why this is a separate pass from the energy-gate fix (embed_stems.py): the
energy gate only catches near-silent stems. It can't catch a stem that's
genuinely non-silent but still isn't voice -- the observed real case is
Demucs mistaking sustained cello/violin melody for singing (similar
frequency/timbral range). AST checks the ORIGINAL MIX, an independent
signal the energy gate has no access to.

Runs entirely on CPU -- no GPU/Colab dependency, unlike the stem-separation
pipeline itself. Checkpointed by song (saves the index + prints progress
every CHECKPOINT_EVERY songs) since a full-library run takes real
wall-clock time -- see scripts/sample_vocal_segment_prevalence.py for the
prevalence estimate this scope was sized against. Safe to re-run: a segment
already 'skipped' is left alone (not re-scored), and a segment whose vector
is already gone from the index is cheap to no-op on a second pass.
"""

from pathlib import Path

import librosa

from sonic_explorer.config import ARTIFACTS_DIR, DB_PATH, audio_path_for
from sonic_explorer.pipeline.vocal_presence import AST_SAMPLE_RATE, has_vocal_content
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

FACET_NAME = "vocal"
CHECKPOINT_EVERY = 50
LOG_PATH = Path(__file__).resolve().parent / "filter_vocal_facet_by_ast_flagged.txt"


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
    embedding_repo.load_index(FACET_NAME)

    all_songs = song_repo.list_songs()
    total = len(all_songs)
    songs_with_segments_checked = 0
    segments_checked = 0
    flagged: list[tuple[str, str, float, float]] = []  # (title, genre, start_sec, end_sec)
    failed: list[tuple[int, str]] = []

    print(f"Scanning {total} songs for vocal-facet segments to check...")

    for i, song in enumerate(all_songs):
        segments = song_repo.get_segments(song.id)
        done_segments = [seg for seg in segments if embedding_repo.status(seg.id, FACET_NAME) == "done"]
        if not done_segments:
            continue

        songs_with_segments_checked += 1
        try:
            audio, sr = librosa.load(str(audio_path_for(song)), sr=AST_SAMPLE_RATE, mono=True)
        except Exception as exc:  # noqa: BLE001 -- one bad file must not kill the whole run
            failed.append((song.fma_track_id, str(exc)))
            continue

        for seg in done_segments:
            start, end = int(seg.start_sec * sr), int(seg.end_sec * sr)
            window = audio[start:end]
            if len(window) < sr * 0.5:
                continue  # degenerate tail window, not enough audio to classify meaningfully

            segments_checked += 1
            try:
                if has_vocal_content(window, sr):
                    continue  # real vocal content confirmed for this specific segment -- keep it
                embedding_repo.remove_from_index(FACET_NAME, seg.id)
                embedding_repo.mark_skipped(seg.id, FACET_NAME)
                flagged.append((song.title, song.genre_top, seg.start_sec, seg.end_sec))
            except Exception as exc:  # noqa: BLE001
                failed.append((song.fma_track_id, str(exc)))

        if songs_with_segments_checked % CHECKPOINT_EVERY == 0:
            embedding_repo.save_index(FACET_NAME)
            print(f"  ...{i + 1}/{total} songs scanned, {songs_with_segments_checked} had vocal-facet "
                  f"segments, {segments_checked} segments checked, {len(flagged)} flagged so far")

    embedding_repo.save_index(FACET_NAME)

    print(f"\nDone. {songs_with_segments_checked} songs / {segments_checked} segments checked.")
    print(f"{len(flagged)} segments flagged (no singing/speech detected) and removed from the vocal index.")
    if failed:
        print(f"{len(failed)} failures (see below):")
        for track_id, err in failed:
            print(f"  track {track_id}: {err}")

    LOG_PATH.write_text(
        "\n".join(f"{title!r} ({genre}) {start:.1f}-{end:.1f}s" for title, genre, start, end in flagged),
        encoding="utf-8",
    )
    print(f"Flagged-segment list written to {LOG_PATH}")


if __name__ == "__main__":
    main()
