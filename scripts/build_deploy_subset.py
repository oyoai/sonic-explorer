"""Builds deploy_data/ -- a small, genre-balanced, stratified subset of the real
local library, committed to git so Streamlit Community Cloud (which only ever
sees what's in the repo -- no Drive, no local disk) has something to run against.
sonic_explorer.config falls back to deploy_data/ automatically whenever data/
(gitignored, the full ~1.4GB local dataset) isn't present.

Rebuilds from scratch every run (wipes deploy_data/ first) -- cheap, since this
only copies already-computed vectors/audio/structure, no CLAP/librosa recompute.
"""

import shutil

import numpy as np

from sonic_explorer.config import PROJECT_ROOT
from sonic_explorer.models import Song
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

SONGS_PER_GENRE = 25  # ~25 x 8 genres = ~200 songs, ~160MB audio -- comfortably small for a git repo
SEED = 42
FACETS = ["sound", "harmony"]

SOURCE_DATA_DIR = PROJECT_ROOT / "data"
DEPLOY_DATA_DIR = PROJECT_ROOT / "deploy_data"


def main():
    if not (SOURCE_DATA_DIR / "artifacts" / "sonic_explorer.db").exists():
        raise SystemExit(f"No real data found at {SOURCE_DATA_DIR} -- nothing to subset from.")

    if DEPLOY_DATA_DIR.exists():
        shutil.rmtree(DEPLOY_DATA_DIR)
    (DEPLOY_DATA_DIR / "audio").mkdir(parents=True)
    (DEPLOY_DATA_DIR / "artifacts" / "structure").mkdir(parents=True)

    src_conn = init_db(SOURCE_DATA_DIR / "artifacts" / "sonic_explorer.db")
    src_song_repo = SongRepository(src_conn)
    src_embedding_repo = EmbeddingRepository(src_conn, artifacts_dir=SOURCE_DATA_DIR / "artifacts")
    for facet_name in FACETS:
        src_embedding_repo.load_index(facet_name)

    dst_conn = init_db(DEPLOY_DATA_DIR / "artifacts" / "sonic_explorer.db")
    dst_song_repo = SongRepository(dst_conn)
    dst_embedding_repo = EmbeddingRepository(dst_conn, artifacts_dir=DEPLOY_DATA_DIR / "artifacts")

    rng = np.random.default_rng(SEED)
    all_songs = src_song_repo.list_songs()
    by_genre: dict[str, list] = {}
    for song in all_songs:
        by_genre.setdefault(song.genre_top, []).append(song)

    sampled = []
    for genre, songs in by_genre.items():
        idx = rng.choice(len(songs), size=min(SONGS_PER_GENRE, len(songs)), replace=False)
        sampled.extend(songs[i] for i in idx)
    print(f"Sampled {len(sampled)} songs across {len(by_genre)} genres")

    n_audio_copied = 0
    n_structure_copied = 0
    n_segments_copied = {facet_name: 0 for facet_name in FACETS}
    n_dna_copied = 0

    for old_song in sampled:
        new_song = Song(
            filepath=old_song.filepath,  # vestigial -- audio_path_for() reconstructs by fma_track_id, ignores this
            fma_track_id=old_song.fma_track_id,
            title=old_song.title,
            artist=old_song.artist,
            genre_top=old_song.genre_top,
            duration_sec=old_song.duration_sec,
        )
        new_song_id = dst_song_repo.add_song(new_song)
        if old_song.tempo_bpm is not None:
            dst_song_repo.update_song_dna(
                new_song_id,
                tempo_bpm=old_song.tempo_bpm,
                energy=old_song.energy,
                brightness=old_song.brightness,
                harmonic_complexity=old_song.harmonic_complexity,
                rhythmic_density=old_song.rhythmic_density,
            )
            n_dna_copied += 1

        old_segments = src_song_repo.get_segments(old_song.id)
        new_segment_ids = dst_song_repo.add_segments(new_song_id, old_segments)

        for old_seg, new_seg_id in zip(old_segments, new_segment_ids):
            for facet_name in FACETS:
                if src_embedding_repo.status(old_seg.id, facet_name) == "done":
                    vector = src_embedding_repo.get_vector(facet_name, old_seg.id)
                    dst_embedding_repo.add_vector(facet_name, new_seg_id, vector)
                    n_segments_copied[facet_name] += 1

        src_audio = SOURCE_DATA_DIR / "audio" / f"{old_song.fma_track_id}.mp3"
        if src_audio.exists():
            shutil.copy(src_audio, DEPLOY_DATA_DIR / "audio" / f"{old_song.fma_track_id}.mp3")
            n_audio_copied += 1

        src_matrix = SOURCE_DATA_DIR / "artifacts" / "structure" / f"{old_song.id}.npy"
        src_timeline = SOURCE_DATA_DIR / "artifacts" / "structure" / f"{old_song.id}_timeline.npz"
        if src_matrix.exists() and src_timeline.exists():
            shutil.copy(src_matrix, DEPLOY_DATA_DIR / "artifacts" / "structure" / f"{new_song_id}.npy")
            shutil.copy(src_timeline, DEPLOY_DATA_DIR / "artifacts" / "structure" / f"{new_song_id}_timeline.npz")
            n_structure_copied += 1

    for facet_name in FACETS:
        dst_embedding_repo.save_index(facet_name)

    print(f"Songs: {len(sampled)}")
    for facet_name in FACETS:
        print(f"Segments with {facet_name} vectors copied: {n_segments_copied[facet_name]}")
    print(f"Audio files copied: {n_audio_copied}")
    print(f"Structure artifacts copied: {n_structure_copied}")
    print(f"Song DNA copied: {n_dna_copied}")
    print(f"Deploy data written to {DEPLOY_DATA_DIR}")


if __name__ == "__main__":
    main()
