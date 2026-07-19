"""Library-scale structure computation (self-similarity matrix + segmented
timeline + sound fingerprint + song DNA). CPU-only (no CLAP/GPU needed) -- can
run locally exactly like scripts/acquire_fma.py, or in Colab reusing whatever
curated audio is already there. Compute-once via checking whether each song's
artifacts already exist (no embedding_status row for this -- these are
song-level artifacts, not per-segment facet vectors; see facets/structure.py).

The sound fingerprint (facets/fingerprint.py) and song DNA (facets/song_dna.py)
are computed here, not lazily in the Streamlit app, specifically so the
deployed app never needs librosa itself -- all three derived artifacts reuse
the one audio load already happening for structure computation, and the
results are just small arrays/scalars read back at display time."""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sonic_explorer.config import STRUCTURE_SR
from sonic_explorer.facets.fingerprint import sound_fingerprint
from sonic_explorer.facets.song_dna import compute_raw_song_dna
from sonic_explorer.facets.structure import analyze_structure
from sonic_explorer.repository.song_repository import SongRepository


def timeline_path(structure_dir: Path, song_id: int) -> Path:
    return structure_dir / f"{song_id}_timeline.npz"


def run_batch_structure(
    manifest_df: pd.DataFrame,
    audio_dir: Path,
    song_repo: SongRepository,
    structure_dir: Path,
    sr: int = STRUCTURE_SR,
    checkpoint_every: int = 50,
    on_checkpoint: Callable[[int, int], None] | None = None,
    on_error: Callable[[int, Exception], None] | None = None,
) -> list[int]:
    """manifest_df needs columns: track_id, relative_path. Song rows are expected
    to already exist (via the sound-facet batch job) -- tracks not yet in the DB
    are skipped rather than creating a new, structure-only song row.

    A single malformed/corrupt audio file (real risk at real-dataset scale --
    truncated downloads, near-empty decodes) must not take down the whole batch:
    failures are isolated per-song, reported via on_error, and retried on the next
    run (no .npy gets written, so the compute-once check naturally revisits them).
    Returns the list of track_ids that failed."""
    import librosa

    structure_dir.mkdir(parents=True, exist_ok=True)
    total = len(manifest_df)
    failed_track_ids = []

    for i, row in enumerate(manifest_df.itertuples()):
        song = song_repo.get_song_by_fma_track_id(int(row.track_id))
        if song is None:
            continue

        matrix_path = structure_dir / f"{song.id}.npy"
        tl_path = timeline_path(structure_dir, song.id)
        has_dna = song.tempo_bpm is not None
        if matrix_path.exists() and tl_path.exists() and has_dna:
            continue

        try:
            filepath = str(audio_dir / row.relative_path)
            audio, loaded_sr = librosa.load(filepath, sr=sr, mono=True)
            analysis = analyze_structure(audio, loaded_sr)
            fingerprint = sound_fingerprint(audio, loaded_sr)
            dna = compute_raw_song_dna(audio, loaded_sr)
            np.save(matrix_path, analysis.matrix)
            np.savez(
                tl_path,
                starts=analysis.segment_starts,
                ends=analysis.segment_ends,
                labels=analysis.segment_labels,
                sound_fp=fingerprint,
            )
            song_repo.update_song_dna(
                song.id,
                tempo_bpm=dna["tempo_bpm"],
                energy=dna["energy"],
                brightness=dna["brightness"],
                harmonic_complexity=dna["harmonic_complexity"],
                rhythmic_density=dna["rhythmic_density"],
            )
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
            failed_track_ids.append(int(row.track_id))
            if on_error:
                on_error(int(row.track_id), exc)

        if on_checkpoint and (i + 1) % checkpoint_every == 0:
            on_checkpoint(i + 1, total)

    return failed_track_ids
