"""Library-scale structure-matrix computation. CPU-only (no CLAP/GPU needed) --
can run locally exactly like scripts/acquire_fma.py, or in Colab reusing whatever
curated audio is already there. Compute-once via checking whether each song's
.npy artifact already exists (no embedding_status row for this -- it's a
song-level artifact, not a per-segment facet vector; see facets/structure.py)."""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sonic_explorer.config import STRUCTURE_SR
from sonic_explorer.facets.structure import compute_self_similarity_matrix
from sonic_explorer.repository.song_repository import SongRepository


def run_batch_structure(
    manifest_df: pd.DataFrame,
    audio_dir: Path,
    song_repo: SongRepository,
    structure_dir: Path,
    sr: int = STRUCTURE_SR,
    checkpoint_every: int = 50,
    on_checkpoint: Callable[[int, int], None] | None = None,
) -> None:
    """manifest_df needs columns: track_id, relative_path. Song rows are expected
    to already exist (via the sound-facet batch job) -- tracks not yet in the DB
    are skipped rather than creating a new, structure-only song row."""
    import librosa

    structure_dir.mkdir(parents=True, exist_ok=True)
    total = len(manifest_df)

    for i, row in enumerate(manifest_df.itertuples()):
        song = song_repo.get_song_by_fma_track_id(int(row.track_id))
        if song is None:
            continue

        out_path = structure_dir / f"{song.id}.npy"
        if out_path.exists():
            continue

        filepath = str(audio_dir / row.relative_path)
        audio, loaded_sr = librosa.load(filepath, sr=sr, mono=True)
        matrix = compute_self_similarity_matrix(audio, loaded_sr)
        np.save(out_path, matrix)

        if on_checkpoint and (i + 1) % checkpoint_every == 0:
            on_checkpoint(i + 1, total)
