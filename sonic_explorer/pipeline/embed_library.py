"""The actual promoted batch job: library-scale version of the single-song embed
loop in notebooks/audio_deep_dive.ipynb (cells 29-30). Compute-once via
EmbeddingRepository.status() -- safe to re-run after a Colab disconnect, and skips
even the audio load (not just the embedding) for songs already fully processed
across every requested facet.

Handles any number of facets from one shared audio load per song -- adding a new
facet later (e.g. harmony alongside sound) means passing it in the `facets` list,
not standing up a second pipeline or reloading audio a second time.

Checkpointing is handled internally (not left to the caller's on_checkpoint
callback) specifically so embedding_status is only ever marked 'done' for vectors
that have actually been persisted to disk via save_index() -- see
EmbeddingRepository.add_to_index() for why that ordering matters."""

from pathlib import Path
from typing import Callable

import pandas as pd

from sonic_explorer.config import CLAP_SR
from sonic_explorer.facets.base import Facet
from sonic_explorer.models import Song
from sonic_explorer.pipeline.segment import segment_song
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository


def run_batch_embedding(
    manifest_df: pd.DataFrame,
    audio_dir: Path,
    song_repo: SongRepository,
    embedding_repo: EmbeddingRepository,
    facets: list[Facet],
    checkpoint_every: int = 50,
    on_checkpoint: Callable[[int, int], None] | None = None,
) -> None:
    """manifest_df needs columns: track_id, genre_top, title, artist, relative_path."""
    import librosa

    total = len(manifest_df)
    pending_confirmation: dict[str, list[tuple[int, int | None]]] = {f.name: [] for f in facets}

    def checkpoint():
        for facet in facets:
            pending = pending_confirmation[facet.name]
            if not pending:
                continue
            embedding_repo.save_index(facet.name)
            for seg_id, dim in pending:
                embedding_repo.mark_done(seg_id, facet.name, vector_store_id=seg_id, dim=dim)
            pending.clear()

    for i, row in enumerate(manifest_df.itertuples()):
        existing = song_repo.get_song_by_fma_track_id(int(row.track_id))
        if existing and existing.segments and all(
            embedding_repo.status(seg.id, facet.name) == "done"
            for seg in existing.segments
            for facet in facets
        ):
            continue  # every facet already embedded for every segment -- skip the audio load entirely

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

        for facet in facets:
            pending = [
                (seg_id, seg)
                for seg_id, seg in zip(seg_ids, segments)
                if embedding_repo.status(seg_id, facet.name) != "done"
            ]
            if pending:
                windows = [audio[int(seg.start_sec * sr):int(seg.end_sec * sr)] for _, seg in pending]
                vectors = facet.embed_batch(windows, sr)
                for (seg_id, _), vector in zip(pending, vectors):
                    embedding_repo.add_to_index(facet.name, seg_id, vector)
                    pending_confirmation[facet.name].append((seg_id, vector.shape[-1]))

        if (i + 1) % checkpoint_every == 0:
            checkpoint()
            if on_checkpoint:
                on_checkpoint(i + 1, total)

    checkpoint()  # final flush -- nothing should stay unconfirmed after the run completes
