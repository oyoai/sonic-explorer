"""Seeds a small local dev dataset -- synthetic audio + fake but genre-clustered
embeddings -- so the interface layer (RetrievalService, Streamlit pages) can be
built and tested without waiting on the real Colab embedding job. Same DB/FAISS
shape and repository contracts as the real pipeline, so swapping in the real
synced artifacts later needs zero code changes -- just overwrite data/artifacts/
and data/audio/ with the real ones.

NOT part of the real pipeline -- dev-only tool, never run against real data.
"""

import numpy as np
import soundfile as sf

from sonic_explorer.config import ARTIFACTS_DIR, AUDIO_DIR, CLAP_SR, DB_PATH, DEV_DATA_MARKER, STRUCTURE_DIR
from sonic_explorer.facets.structure import compute_self_similarity_matrix
from sonic_explorer.models import Song
from sonic_explorer.pipeline.segment import segment_song
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

GENRES = ["Rock", "Jazz", "Electronic", "Folk"]
BASE_FREQ = {"Rock": 220.0, "Jazz": 330.0, "Electronic": 440.0, "Folk": 550.0}
SONGS_PER_GENRE = 8
SONG_DURATION_SEC = 20.0
EMBED_DIM = 32

rng = np.random.default_rng(42)


def make_song_audio(path, freq, duration_sec=SONG_DURATION_SEC, sr=CLAP_SR):
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    audio = (
        0.15 * np.sin(2 * np.pi * freq * t)
        + 0.05 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.01 * rng.standard_normal(len(t))
    ).astype(np.float32)
    sf.write(str(path), audio, sr)
    return audio


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)

    conn = init_db(DB_PATH)
    song_repo = SongRepository(conn)
    embedding_repo = EmbeddingRepository(conn, artifacts_dir=ARTIFACTS_DIR)

    # a distinct embedding "center" per genre so PCA + K-means on the Taste Map
    # actually shows separated clusters, instead of undifferentiated noise
    genre_centers = {genre: rng.normal(size=EMBED_DIM) * 3 for genre in GENRES}

    track_id = 1
    for genre in GENRES:
        for i in range(SONGS_PER_GENRE):
            filename = f"dev_{track_id}.wav"
            filepath = AUDIO_DIR / filename
            audio = make_song_audio(filepath, BASE_FREQ[genre] * (1 + 0.05 * i))

            song = Song(
                filepath=str(filepath),
                fma_track_id=track_id,
                title=f"{genre} Song {i + 1}",
                artist=f"{genre} Artist",
                genre_top=genre,
                duration_sec=SONG_DURATION_SEC,
            )
            song_id = song_repo.add_song(song)
            segments = segment_song(song_id, SONG_DURATION_SEC)
            seg_ids = song_repo.add_segments(song_id, segments)

            for seg_id in seg_ids:
                vec = genre_centers[genre] + rng.normal(size=EMBED_DIM) * 0.5
                embedding_repo.add_vector("sound", seg_id, vec.astype(np.float32))

            ssm = compute_self_similarity_matrix(audio, CLAP_SR)
            np.save(STRUCTURE_DIR / f"{song_id}.npy", ssm)

            print(f"  [{track_id}/{len(GENRES) * SONGS_PER_GENRE}] {song.title}")
            track_id += 1

    embedding_repo.save_index("sound")
    DEV_DATA_MARKER.write_text("Synthetic dev data seeded by scripts/seed_dev_data.py -- delete this file "
                                "(or overwrite data/artifacts/ with the real sync) once real data lands.\n")
    print(f"Seeded {track_id - 1} songs across {len(GENRES)} genres")
    print(f"DB: {DB_PATH}")
    print(f"Audio: {AUDIO_DIR}")
    print(f"Structure matrices: {STRUCTURE_DIR}")
    print(f"Sound index size: {embedding_repo.index_size('sound')}")


if __name__ == "__main__":
    main()
