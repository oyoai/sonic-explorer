"""One-off portability fix: song.filepath in the DB was set to the Colab-side
Drive-mounted path (e.g. /content/drive/MyDrive/SonicExplorer/fma_curated/audio/x.mp3)
during the batch embedding run in notebooks/02_batch_embed_pipeline.ipynb. Once the
real curated audio is synced down into this machine's data/audio/ (named
{fma_track_id}.mp3, per notebooks/01's extraction convention), this repoints every
song's filepath to the local path so st.audio() and friends actually resolve.

Safe to re-run -- only updates rows whose filepath doesn't already match.
"""

from sonic_explorer.config import AUDIO_DIR, ARTIFACTS_DIR, DB_PATH
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.song_repository import SongRepository


def main():
    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)

    updated = 0
    missing = []
    for song in song_repo.list_songs():
        expected_path = str(AUDIO_DIR / f"{song.fma_track_id}.mp3")
        if not (AUDIO_DIR / f"{song.fma_track_id}.mp3").exists():
            missing.append(song.fma_track_id)
        if song.filepath != expected_path:
            song_repo.update_filepath(song.id, expected_path)
            updated += 1

    print(f"Repathed {updated} songs to point at {AUDIO_DIR}")
    if missing:
        print(f"WARNING: {len(missing)} songs have no matching audio file in {AUDIO_DIR} "
              f"(e.g. track_id {missing[0]}) -- check the audio sync completed.")


if __name__ == "__main__":
    main()
