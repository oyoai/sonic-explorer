"""Runs the structure-matrix batch job against whatever's currently in the DB +
data/audio/. Builds the manifest straight from the songs table (track_id,
relative_path) -- no need for the original curated_tracks.csv locally, since
every song's filepath already resolves to data/audio/{fma_track_id}.mp3 (see
repath_audio_paths.py). CPU-only, safe to re-run (compute-once via file existence).
"""

import pandas as pd

from sonic_explorer.config import ARTIFACTS_DIR, AUDIO_DIR, DB_PATH, STRUCTURE_DIR
from sonic_explorer.pipeline.build_structure_library import run_batch_structure
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.song_repository import SongRepository


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)

    songs = song_repo.list_songs()
    manifest = pd.DataFrame([
        {"track_id": s.fma_track_id, "relative_path": f"{s.fma_track_id}.mp3"} for s in songs
    ])
    print(f"{len(manifest)} songs to process")

    def on_checkpoint(done, total):
        print(f"  [{done}/{total}] structure matrices computed")

    def on_error(track_id, exc):
        print(f"  WARNING: track {track_id} failed ({type(exc).__name__}: {exc}) -- skipped, will retry next run")

    failed = run_batch_structure(
        manifest_df=manifest,
        audio_dir=AUDIO_DIR,
        song_repo=song_repo,
        structure_dir=STRUCTURE_DIR,
        checkpoint_every=50,
        on_checkpoint=on_checkpoint,
        on_error=on_error,
    )

    n_matrices = len(list(STRUCTURE_DIR.glob("*.npy")))
    print(f"Done. {n_matrices} structure matrices in {STRUCTURE_DIR}")
    if failed:
        print(f"{len(failed)} track(s) failed and were skipped: {failed}")


if __name__ == "__main__":
    main()
