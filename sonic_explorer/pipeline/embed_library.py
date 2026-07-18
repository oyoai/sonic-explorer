"""The actual promoted batch job: library-scale version of the single-song embed
loop in notebooks/audio_deep_dive.ipynb (cells 29-30). Compute-once via
EmbeddingRepository.status() -- safe to re-run after a Colab disconnect, and skips
even the audio load (not just the embedding) for songs already fully processed."""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sonic_explorer.config import CLAP_SR
from sonic_explorer.facets.sound import SoundFacet
from sonic_explorer.models import Song
from sonic_explorer.pipeline.segment import segment_song
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository


def run_batch_embedding(
    manifest_df: pd.DataFrame,
    audio_dir: Path,
    song_repo: SongRepository,
    embedding_repo: EmbeddingRepository,
    sound_facet: SoundFacet,
    facet_name: str = "sound",
    checkpoint_every: int = 50,
    on_checkpoint: Callable[[int, int], None] | None = None,
) -> None:
    """manifest_df needs columns: track_id, genre_top, title, artist, relative_path."""
    import librosa

    total = len(manifest_df)
    for i, row in enumerate(manifest_df.itertuples()):
        existing = song_repo.get_song_by_fma_track_id(int(row.track_id))
        if existing and existing.segments and all(
            embedding_repo.status(seg.id, facet_name) == "done" for seg in existing.segments
        ):
            continue  # fully embedded already -- skip the audio load entirely

        filepath = str(audio_dir / row.relative_path)
        audio, sr = librosa.load(filepath, sr=CLAP_SR, mono=True)
        duration_sec = len(audio) / sr

        song = Song(
            filepath=filepath,
            fma_track_id=int(row.track_id),
            title=row.title,
            artist=row.artist,
            genre_top=row.genre_top,
            duration_sec=duration_sec,
        )
        song_id = song_repo.add_song(song)

        segments = segment_song(song_id, duration_sec)
        seg_ids = song_repo.add_segments(song_id, segments)

        pending = [
            (seg_id, seg)
            for seg_id, seg in zip(seg_ids, segments)
            if embedding_repo.status(seg_id, facet_name) != "done"
        ]
        if pending:
            windows = [audio[int(seg.start_sec * sr):int(seg.end_sec * sr)] for _, seg in pending]
            vectors = sound_facet.embed_batch(windows, sr)
            for (seg_id, _), vector in zip(pending, vectors):
                embedding_repo.add_vector(facet_name, seg_id, vector)

        if on_checkpoint and (i + 1) % checkpoint_every == 0:
            on_checkpoint(i + 1, total)
