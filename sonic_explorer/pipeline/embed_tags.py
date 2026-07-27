"""Library-scale sound-tags facet embedding: for each song's already-existing
segments, run AST tagging directly on the ORIGINAL MIX audio (not a stem --
there's no separation step here, unlike embed_stems.py) per segment, then
embed the resulting tag text via CLAP's text encoder (facets/tags.py's
SoundTagsFacet). Reuses the same compute-once/checkpoint discipline as every
other batch pipeline in this package.

A separate module from embed_library.py/embed_stems.py rather than a
generalization of either: this facet's embed() does real per-segment AST
inference (not a lightweight CLAP-audio call), and its "gate" is a raised
NoTagsDetected from inside embed() itself rather than a separate energy-ratio
check against the stem -- different enough shape to not force into either
existing pipeline function.

Song rows (and their segments) are expected to already exist -- same
precondition as embed_stems.py, since this only makes sense to run after the
sound-facet batch job has already populated the library."""

from pathlib import Path
from typing import Callable

import pandas as pd

from sonic_explorer.facets.tags import NoTagsDetected, SoundTagsFacet
from sonic_explorer.pipeline.sound_tagging import AST_SAMPLE_RATE
from sonic_explorer.repository.embedding_repository import EmbeddingRepository
from sonic_explorer.repository.song_repository import SongRepository

_FINISHED_STATUSES = ("done", "skipped")


def run_batch_tags_embedding(
    manifest_df: pd.DataFrame,
    audio_dir: Path,
    song_repo: SongRepository,
    embedding_repo: EmbeddingRepository,
    facet: SoundTagsFacet,
    checkpoint_every: int = 25,
    on_checkpoint: Callable[[int, int], None] | None = None,
    on_error: Callable[[int, Exception], None] | None = None,
) -> list[int]:
    """manifest_df needs columns: track_id, relative_path. Returns the list
    of track_ids that failed (isolated, not fatal -- same discipline as
    embed_stems.py: one bad file must not lose progress on the rest)."""
    import librosa

    total = len(manifest_df)
    failed_track_ids: list[int] = []
    pending_confirmation: list[tuple[int, int | None]] = []

    def checkpoint():
        if not pending_confirmation:
            return
        embedding_repo.save_index(facet.name)
        for seg_id, dim in pending_confirmation:
            embedding_repo.mark_done(seg_id, facet.name, vector_store_id=seg_id, dim=dim)
        pending_confirmation.clear()

    for i, row in enumerate(manifest_df.itertuples()):
        song = song_repo.get_song_by_fma_track_id(int(row.track_id))
        if song is None or not song.segments:
            continue

        pending_segments = [
            seg for seg in song.segments if embedding_repo.status(seg.id, facet.name) not in _FINISHED_STATUSES
        ]
        if not pending_segments:
            continue  # every segment already resolved (embedded or skipped) for this facet

        try:
            filepath = str(audio_dir / row.relative_path)
            audio, sr = librosa.load(filepath, sr=AST_SAMPLE_RATE, mono=True)

            for seg in pending_segments:
                start, end = int(seg.start_sec * sr), int(seg.end_sec * sr)
                window = audio[start:end]
                if len(window) < sr * 0.5:
                    continue  # degenerate tail window -- left 'pending', not worth a real verdict either way

                try:
                    vector = facet.embed(window, sr)
                except NoTagsDetected:
                    embedding_repo.mark_skipped(seg.id, facet.name)
                    continue

                embedding_repo.add_to_index(facet.name, seg.id, vector)
                pending_confirmation.append((seg.id, vector.shape[-1]))
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
            failed_track_ids.append(int(row.track_id))
            if on_error:
                on_error(int(row.track_id), exc)

        if (i + 1) % checkpoint_every == 0:
            checkpoint()
            if on_checkpoint:
                on_checkpoint(i + 1, total)

    checkpoint()  # final flush -- nothing should stay unconfirmed after the run completes
    return failed_track_ids
