"""Runs the harmony-facet batch job against whatever's currently in the DB +
data/audio/. CPU-only (chroma extraction, no CLAP/GPU needed) -- unlike the sound
facet, this never needs a Colab round-trip, so it's run straight from local dev.

Builds the manifest straight from the songs table -- every song is already fully
populated (title/artist/genre_top) from the sound-facet pass, and
run_batch_embedding's per-facet compute-once check means the sound facet is
skipped entirely (already 'done' for every segment) while harmony gets embedded
from the same shared audio load.
"""

import pandas as pd

from sonic_explorer.config import ARTIFACTS_DIR, AUDIO_DIR, DB_PATH
from sonic_explorer.facets.harmony import HarmonyFacet
from sonic_explorer.pipeline.embed_library import run_batch_embedding
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)
    embedding_repo.load_index("harmony")

    songs = song_repo.list_songs()
    manifest = pd.DataFrame([
        {
            "track_id": s.fma_track_id,
            "genre_top": s.genre_top,
            "title": s.title,
            "artist": s.artist,
            "relative_path": f"{s.fma_track_id}.mp3",
        }
        for s in songs
    ])
    print(f"{len(manifest)} songs to process")
    print(f"Resuming with existing harmony index size: {embedding_repo.index_size('harmony')}")

    harmony_facet = HarmonyFacet()

    def on_checkpoint(done, total):
        print(f"  [{done}/{total}] checkpoint -- harmony index now has {embedding_repo.index_size('harmony')} vectors")

    run_batch_embedding(
        manifest_df=manifest,
        audio_dir=AUDIO_DIR,
        song_repo=song_repo,
        embedding_repo=embedding_repo,
        facets=[harmony_facet],
        checkpoint_every=50,
        on_checkpoint=on_checkpoint,
    )

    print("Batch embedding complete.")
    print(f"Final harmony index size: {embedding_repo.index_size('harmony')}")
    print(f"Total songs in DB: {len(song_repo.list_songs())}")


if __name__ == "__main__":
    main()
