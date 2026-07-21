"""One-off filter pass over the vocal facet: for every song with 'done'
vocal-facet segments, run a pretrained AudioSet tagger (AST) against the
song's real mix audio to check whether it actually contains singing/speech
at all (pipeline/vocal_presence.py). Songs that fail the check have their
vocal-facet segments removed from the FAISS index and marked 'skipped'.

Why this is a separate pass from the energy-gate fix (embed_stems.py): the
energy gate only catches near-silent stems. It can't catch a stem that's
genuinely non-silent but still isn't voice -- the observed real case is
Demucs mistaking sustained cello/violin melody for singing (similar
frequency/timbral range). AST checks the ORIGINAL MIX, an independent
signal the energy gate has no access to.

Runs entirely on CPU -- no GPU/Colab dependency, unlike the stem-separation
pipeline itself. Checkpointed (saves the index + prints progress every
CHECKPOINT_EVERY songs) since a full-library run takes real wall-clock time.
Safe to re-run: songs already 'skipped' are left alone, and a song whose
vocal segments are already gone from the index is cheap to re-check and
just does nothing on the second pass.
"""

import time
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
    checked = 0
    flagged_songs: list[tuple[str, str, int]] = []  # (title, genre, n_segments_removed)
    failed: list[tuple[int, str]] = []

    print(f"Checking {total} songs for vocal-facet segments to filter...")

    for i, song in enumerate(all_songs):
        segments = song_repo.get_segments(song.id)
        done_segments = [seg for seg in segments if embedding_repo.status(seg.id, FACET_NAME) == "done"]
        if not done_segments:
            continue

        checked += 1
        try:
            audio, sr = librosa.load(str(audio_path_for(song)), sr=AST_SAMPLE_RATE, mono=True)
            if has_vocal_content(audio, sr):
                continue  # real vocal content confirmed -- nothing to filter

            for seg in done_segments:
                embedding_repo.remove_from_index(FACET_NAME, seg.id)
                embedding_repo.mark_skipped(seg.id, FACET_NAME)
            flagged_songs.append((song.title, song.genre_top, len(done_segments)))
        except Exception as exc:  # noqa: BLE001 -- one bad file must not kill the whole run
            failed.append((song.fma_track_id, str(exc)))

        if checked % CHECKPOINT_EVERY == 0:
            embedding_repo.save_index(FACET_NAME)
            print(f"  ...{i + 1}/{total} songs scanned, {checked} had vocal-facet segments, "
                  f"{len(flagged_songs)} flagged so far")

    embedding_repo.save_index(FACET_NAME)

    print(f"\nDone. {checked} songs had vocal-facet segments to check.")
    print(f"{len(flagged_songs)} songs flagged (no singing/speech detected) -- "
          f"{sum(n for _, _, n in flagged_songs)} segments removed from the vocal index.")
    if failed:
        print(f"{len(failed)} songs failed to process (see printed errors above/below):")
        for track_id, err in failed:
            print(f"  track {track_id}: {err}")

    LOG_PATH.write_text(
        "\n".join(f"{title!r} ({genre}) -- {n} segment(s) removed" for title, genre, n in flagged_songs),
        encoding="utf-8",
    )
    print(f"Flagged-song list written to {LOG_PATH}")


if __name__ == "__main__":
    main()
