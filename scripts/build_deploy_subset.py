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
from sonic_explorer.facets.registry import default_registry
from sonic_explorer.models import Song
from sonic_explorer.repository.db import init_db
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

SONGS_PER_GENRE = 25  # ~25 x 8 genres = ~200 songs, ~160MB audio -- comfortably small for a git repo
SEED = 42
FACETS = default_registry().names()  # sound, harmony, vocal, drums, bass, instrumental

SOURCE_DATA_DIR = PROJECT_ROOT / "data"
DEPLOY_DATA_DIR = PROJECT_ROOT / "deploy_data"

# Songs whose exact identity matters to hardcoded evidence in Methodology's
# 5b (real precomputed similarity scores + LLM-generated explanations tied
# to these specific pairs -- see NN_EXAMPLES in
# streamlit_app/pages/0_Methodology.py). Unlike that page's 3a/3b examples
# (made dataset-size-agnostic instead, since any song works equally well
# there), 5b's examples can't be dynamically substituted without regenerating
# real explanations, so they're force-included here regardless of the random
# stratified sample -- otherwise the deployed subset silently drops the
# audio player for whichever of these the random draw missed (confirmed:
# 23 of 24 were missing before this existed). Kept as a plain literal list
# rather than imported from 0_Methodology.py, which isn't import-safe (its
# module-level code calls Streamlit commands that need a real script run) --
# if NN_EXAMPLES changes there, update this list too.
REQUIRED_EXAMPLE_TITLES = [
    "Terminally in Love With You", "Ave",
    "Elektra (You Were Such Fun)", "Mr. Person",
    "Ordinary Girl", "Plasma",
    "Mad Honey", "This is based upon a true story",
    "A Friendly Noose", "300 Days In July",
    "something brewing", "Unless",
    "Lovedropper", "western chow yun-fat",
    "It's Okay, Roseanne", "Sillable",
    "The Drop (Gung Who Version)", "The Beast Is A Computer In Luxemburg",
    "Inspiration", "All I Am",
    "Sam's Song", "Spot Rockers",
    "Squinting at the Sun (radio edit)", "Do Easy",
]


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
    for _genre, songs in by_genre.items():
        idx = rng.choice(len(songs), size=min(SONGS_PER_GENRE, len(songs)), replace=False)
        sampled.extend(songs[i] for i in idx)
    print(f"Sampled {len(sampled)} songs across {len(by_genre)} genres")

    sampled_ids = {s.id for s in sampled}
    songs_by_title = {s.title: s for s in all_songs}
    n_added, n_missing = 0, 0
    for title in REQUIRED_EXAMPLE_TITLES:
        song = songs_by_title.get(title)
        if song is None:
            print(f"WARNING: required example song {title!r} not found in source library at all -- skipping")
            n_missing += 1
        elif song.id not in sampled_ids:
            sampled.append(song)
            sampled_ids.add(song.id)
            n_added += 1
    print(f"Force-included {n_added} required example songs not already in the sample "
          f"({n_missing} missing from the source library entirely) -- {len(sampled)} songs total")

    n_audio_copied = 0
    n_structure_copied = 0
    n_segments_copied = {facet_name: 0 for facet_name in FACETS}
    n_dna_copied = 0
    n_description_copied = 0
    n_sound_tags_copied = 0
    n_metadata_extras_copied = 0

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
        if old_song.description:
            dst_song_repo.update_description(new_song_id, old_song.description)
            n_description_copied += 1
        if old_song.sound_tags:
            dst_song_repo.update_sound_tags(new_song_id, old_song.sound_tags)
            n_sound_tags_copied += 1
        if old_song.genres_all or old_song.album_id or old_song.track_tags:
            dst_song_repo.update_metadata_extras(
                new_song_id,
                genres_all=old_song.genres_all,
                album_id=old_song.album_id,
                album_title=old_song.album_title,
                track_tags=old_song.track_tags,
            )
            n_metadata_extras_copied += 1

        old_segments = src_song_repo.get_segments(old_song.id)
        new_segment_ids = dst_song_repo.add_segments(new_song_id, old_segments)

        for old_seg, new_seg_id in zip(old_segments, new_segment_ids, strict=False):
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
    print(f"Descriptions copied: {n_description_copied}")
    print(f"Sound tags copied: {n_sound_tags_copied}")
    print(f"FMA metadata extras (genres_all/album/tags) copied: {n_metadata_extras_copied}")
    print(f"Deploy data written to {DEPLOY_DATA_DIR}")


if __name__ == "__main__":
    main()
